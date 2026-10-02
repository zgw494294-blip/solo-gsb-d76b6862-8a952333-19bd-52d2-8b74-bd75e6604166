"""可领取审阅工作副本：SQLite 仓库测试（持久化、重启保留、并发串行化、旧库迁移）。"""

import sqlite3
import threading

import pytest

from app.copies import (
    AnnotationNotFound,
    CopyFinalizedError,
    CopyNotFound,
    CopyStore,
    CopyValidationError,
    GroupHeldError,
    HoldersExistError,
    ModifyConflictError,
    RevisionConflictError,
)

TEXT = "x" * 30


def ann(id, start, end, label="L"):
    return {"id": id, "start": start, "end": end, "label": label}


@pytest.fixture()
def store(tmp_path):
    return CopyStore(tmp_path / "copies.db")


def _ids_holders(snap):
    return {a["id"]: a["holder"] for a in snap["annotations"]}


# ---------- 基本持久化 ----------

def test_create_returns_initial_snapshot(store):
    snap = store.create(TEXT, [ann("a", 0, 3), ann("b", 2, 5)])
    assert len(snap["id"]) == 32
    assert snap["revision"] == 1
    assert _ids_holders(snap) == {"a": None, "b": None}
    assert snap["groups"] == [{"annotation_ids": ["a", "b"], "holder": None}]


def test_reopen_store_retains_copy(tmp_path):
    db = tmp_path / "copies.db"
    s1 = CopyStore(db)
    snap = s1.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7), ann("c", 15, 18)])
    cid = snap["id"]
    s1.claim(cid, "a", "alice")
    s1.modify(cid, "a", "alice", 0, 2, "新")

    # 模拟容器重启：用同一数据库文件重新打开仓库
    s2 = CopyStore(db)
    again = s2.get(cid)
    assert again["text"] == TEXT
    assert again["revision"] == 3  # 领取 + 修改
    assert _ids_holders(again) == {"a": "alice", "b": "alice", "c": None}
    changed = [a for a in again["annotations"] if a["id"] == "a"][0]
    assert (changed["start"], changed["end"], changed["label"]) == (0, 2, "新")
    assert again["groups"] == [{"annotation_ids": ["a"], "holder": "alice"},
                               {"annotation_ids": ["b"], "holder": "alice"},
                               {"annotation_ids": ["c"], "holder": None}]


def test_schema_init_is_idempotent(tmp_path):
    db = tmp_path / "copies.db"
    CopyStore(db)
    CopyStore(db)  # 再次初始化不报错、不破坏数据


def test_get_unknown_copy_404(store):
    with pytest.raises(CopyNotFound):
        store.get("deadbeef")


def test_claim_unknown_copy_404(store):
    with pytest.raises(CopyNotFound):
        store.claim("deadbeef", "a", "alice")


def test_claim_unknown_annotation_404(store):
    cid = store.create(TEXT, [])["id"]
    with pytest.raises(AnnotationNotFound):
        store.claim(cid, "nope", "alice")


def test_failed_claim_persists_nothing(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    store.claim(cid, "a", "alice")
    with pytest.raises(GroupHeldError):
        store.claim(cid, "b", "bob")
    snap = store.get(cid)
    assert _ids_holders(snap) == {"a": "alice", "b": "alice"}
    assert snap["revision"] == 2


def test_failed_modify_is_fully_rolled_back(store):
    cid = store.create(TEXT, [ann("a", 0, 2), ann("b", 4, 7)])["id"]
    store.claim(cid, "a", "alice")
    store.claim(cid, "b", "bob")
    with pytest.raises(ModifyConflictError):
        store.modify(cid, "a", "alice", 0, 5, "X")
    snap = store.get(cid)
    a = [a for a in snap["annotations"] if a["id"] == "a"][0]
    assert (a["start"], a["end"], a["label"]) == (0, 2, "L")
    assert _ids_holders(snap) == {"a": "alice", "b": "bob"}
    assert snap["revision"] == 3  # 两次领取，失败修改不增版本


def test_release_persisted(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    store.claim(cid, "a", "alice")
    store.release(cid, "b", "alice")
    snap = store.get(cid)
    assert all(h is None for h in _ids_holders(snap).values())
    assert snap["revision"] == 3


# ---------- 结稿：持久化 / 重启保留 / 失败原子性 ----------

def test_create_and_get_carry_open_status(store):
    snap = store.create(TEXT, [ann("a", 0, 3)])
    assert snap["status"] == "open"
    assert store.get(snap["id"])["status"] == "open"


def test_finalize_persists_status_and_revision(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    snap = store.finalize(cid, 1)
    assert snap["status"] == "finalized" and snap["revision"] == 2
    assert _ids_holders(snap) == {"a": None, "b": None}
    assert store.get(cid)["status"] == "finalized"


def test_finalize_status_retained_across_reopen(tmp_path):
    db = tmp_path / "copies.db"
    s1 = CopyStore(db)
    cid = s1.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    s1.claim(cid, "a", "alice")
    s1.release(cid, "b", "alice")  # revision 1→2→3
    assert s1.finalize(cid, 3)["revision"] == 4

    # 模拟容器重启：结稿状态与修订号跨重启保留
    s2 = CopyStore(db)
    snap = s2.get(cid)
    assert snap["status"] == "finalized" and snap["revision"] == 4
    # 结稿后领取 / 修改 / 再次结稿均 409
    with pytest.raises(CopyFinalizedError):
        s2.claim(cid, "a", "bob")
    with pytest.raises(CopyFinalizedError):
        s2.modify(cid, "a", "bob", 0, 1, "X")
    with pytest.raises(CopyFinalizedError):
        s2.finalize(cid, 4)


def test_finalize_stale_revision_persists_nothing(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    store.claim(cid, "a", "alice")
    store.release(cid, "b", "alice")  # revision=3
    with pytest.raises(RevisionConflictError) as exc:
        store.finalize(cid, 2)
    assert exc.value.current == 3
    snap = store.get(cid)
    assert snap["status"] == "open" and snap["revision"] == 3


def test_finalize_with_holder_persists_nothing(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    store.claim(cid, "a", "alice")  # revision=2
    with pytest.raises(HoldersExistError):
        store.finalize(cid, 2)
    snap = store.get(cid)
    assert snap["status"] == "open"
    assert _ids_holders(snap) == {"a": "alice"}
    assert snap["revision"] == 2


def test_finalize_unknown_copy_404(store):
    with pytest.raises(CopyNotFound):
        store.finalize("deadbeef", 1)


def test_finalize_bad_revision_is_422(store):
    cid = store.create(TEXT, [ann("a", 0, 3)])["id"]
    for bad in (0, -1, True, False, None):
        with pytest.raises(CopyValidationError):
            store.finalize(cid, bad)
    assert store.get(cid)["revision"] == 1


def test_old_schema_without_status_column_migrates(tmp_path):
    # 手工构造旧版数据库：copies 表没有 status 列
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
    CREATE TABLE copies (id TEXT PRIMARY KEY, text TEXT NOT NULL,
                         revision INTEGER NOT NULL DEFAULT 1);
    CREATE TABLE annotations (
        copy_id TEXT NOT NULL REFERENCES copies(id) ON DELETE CASCADE,
        ann_id TEXT NOT NULL, pos INTEGER NOT NULL, start INTEGER NOT NULL,
        end INTEGER NOT NULL, label TEXT NOT NULL, holder TEXT,
        PRIMARY KEY (copy_id, ann_id));
    CREATE INDEX idx_annotations_copy_pos ON annotations(copy_id, pos);
    """)
    conn.execute("INSERT INTO copies(id,text,revision) VALUES('old','文本xyz',3)")
    conn.execute("INSERT INTO annotations VALUES('old','a',0,0,2,'旧',NULL)")
    conn.commit()
    conn.close()

    store = CopyStore(db)  # 启动自动补状态列
    snap = store.get("old")
    assert snap["status"] == "open"          # 旧副本默认 open
    assert snap["revision"] == 3             # 原 revision 不变
    assert snap["text"] == "文本xyz"
    assert snap["annotations"][0]["label"] == "旧"

    # 迁移幂等：再次打开不报错、状态不丢
    assert CopyStore(db).get("old")["status"] == "open"

    # 旧副本沿用原行为，可正常结稿并跨重启保留
    assert store.finalize("old", 3)["status"] == "finalized"
    assert CopyStore(db).get("old")["status"] == "finalized"


def test_old_rows_get_open_default_even_without_migration_path(tmp_path):
    # 经过新版 store 建库后，直接用 SQL 插入省略 status 的行（DEFAULT 生效）
    db = tmp_path / "copies.db"
    store = CopyStore(db)
    cid = store.create(TEXT, [])["id"]
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO copies(id,text,revision) VALUES('raw','q',7)")
    conn.commit()
    conn.close()
    snap = CopyStore(db).get("raw")
    assert snap["status"] == "open" and snap["revision"] == 7


# ---------- 并发：等价于逐次完整执行 ----------

def test_concurrent_claims_serialized_exactly_one_winner(store):
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    errors = []

    def claim(reviewer, barrier):
        barrier.wait()
        try:
            store.claim(cid, "a", reviewer)
        except GroupHeldError as exc:
            errors.append((reviewer, exc.holder))

    n = 16
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=claim, args=(f"r{i}", barrier))
               for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == n - 1  # 恰有一个领取成功
    snap = store.get(cid)
    holders = set(_ids_holders(snap).values())
    assert len(holders) == 1
    assert next(iter(holders)) in {f"r{i}" for i in range(n)}
    assert all(holder == errors[0][1] for holder in holders)
    assert snap["revision"] == 2  # 其余失败请求均不写库，版本只增一次


def test_concurrent_independent_annotation_edits_all_survive(store):
    # 互不重叠的批注由不同审阅者并行修改：任何修改都不得丢失
    anns = [ann(f"a{i}", 3 * i + 0, 3 * i + 2, "L") for i in range(10)]
    cid = store.create(TEXT, anns)["id"]
    for i in range(10):
        store.claim(cid, f"a{i}", f"r{i}")
    base_rev = store.get(cid)["revision"]

    barrier = threading.Barrier(10)

    def edit(i):
        barrier.wait()
        s, e = 3 * i, 3 * i + 1  # 各自缩短，仍互不重叠 / 不相接到他人
        store.modify(cid, f"a{i}", f"r{i}", s, e, f"lab{i}")

    threads = [threading.Thread(target=edit, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = store.get(cid)
    assert snap["revision"] == base_rev + 10
    by_id = {a["id"]: a for a in snap["annotations"]}
    for i in range(10):
        a = by_id[f"a{i}"]
        assert (a["start"], a["end"], a["label"], a["holder"]) == \
            (3 * i, 3 * i + 1, f"lab{i}", f"r{i}")


def test_concurrent_modify_race_into_other_holder(store):
    # a、b 初始相距不连边，分属 alice/bob；两人同时把范围向中间扩展到
    # 重叠区：任一时刻必有人先成功、另一人看到他人持有 → 409 且结果自洽
    cid = store.create(TEXT, [ann("a", 0, 1), ann("b", 5, 6)])["id"]
    store.claim(cid, "a", "alice")
    store.claim(cid, "b", "bob")
    outcomes = {"ok": [], "conflict": []}
    barrier = threading.Barrier(2)

    def grow(ann_id, reviewer, bucket):
        barrier.wait()
        try:
            store.modify(cid, ann_id, reviewer, 0, 4, "L") if ann_id == "a" \
                else store.modify(cid, ann_id, reviewer, 2, 6, "L")
            outcomes["ok"].append(bucket)
        except ModifyConflictError:
            outcomes["conflict"].append(bucket)

    t1 = threading.Thread(target=grow, args=("a", "alice", "a"))
    t2 = threading.Thread(target=grow, args=("b", "bob", "b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(outcomes["ok"]) == 1 and len(outcomes["conflict"]) == 1
    snap = store.get(cid)
    holders = _ids_holders(snap)
    # 冲突失败方的范围与持有权保持修改前状态
    loser = outcomes["conflict"][0]
    a = {x["id"]: x for x in snap["annotations"]}[loser]
    assert (a["start"], a["end"]) == ((0, 1) if loser == "a" else (5, 6))
    assert holders[loser] == ("alice" if loser == "a" else "bob")


def test_snapshot_groups_consistent_after_reopen(tmp_path):
    db = tmp_path / "copies.db"
    s = CopyStore(db)
    cid = s.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7), ann("c", 12, 15)])["id"]
    s.claim(cid, "b", "alice")
    snap = CopyStore(db).get(cid)
    assert snap["groups"] == [
        {"annotation_ids": ["a", "b"], "holder": "alice"},
        {"annotation_ids": ["c"], "holder": None},
    ]


# ---------- 并发：结稿与变更操作等价于某个先后次序 ----------

def test_concurrent_finalize_and_claim_linearizable(store):
    # 无人持有的 open 副本（revision=1）：一个 finalize(expected=1) 与四个
    # 不同审阅者的领取同时发出。可线性化的结局只有两种：
    #  A. 结稿排第一 → 成功（status=finalized, revision=2），其后所有领取
    #     一律 CopyFinalizedError（绝不允许结稿成功后仍写入领取）；
    #  B. 某个领取排第一 → 成功并把 revision 推到 2，结稿必版本过期
    #     （RevisionConflictError），其余领取全部 GroupHeldError，副本仍 open。
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    outcomes = {"finalized": None, "claims": []}
    barrier = threading.Barrier(5)

    def do_finalize():
        barrier.wait()
        try:
            snap = store.finalize(cid, 1)
            outcomes["finalized"] = ("ok", snap["status"], snap["revision"])
        except (RevisionConflictError, HoldersExistError, CopyFinalizedError) as exc:
            outcomes["finalized"] = ("conflict", type(exc).__name__)

    def do_claim(reviewer):
        barrier.wait()
        try:
            store.claim(cid, "a", reviewer)
            outcomes["claims"].append((reviewer, "ok"))
        except GroupHeldError:
            outcomes["claims"].append((reviewer, "held"))
        except CopyFinalizedError:
            outcomes["claims"].append((reviewer, "finalized"))

    threads = [threading.Thread(target=do_finalize)]
    threads += [threading.Thread(target=do_claim, args=(f"r{i}",))
                for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = store.get(cid)
    winners = [r for r, res in outcomes["claims"] if res == "ok"]
    if outcomes["finalized"][0] == "ok":
        # 结局 A：结稿先成功
        assert snap["status"] == "finalized" and snap["revision"] == 2
        assert winners == []
        assert all(res == "finalized" for _, res in outcomes["claims"])
        assert all(h is None for h in _ids_holders(snap).values())
    else:
        # 结局 B：领取先成功，结稿版本过期，副本仍 open
        assert outcomes["finalized"][1] == "RevisionConflictError"
        assert snap["status"] == "open" and snap["revision"] == 2
        assert len(winners) == 1
        assert {res for _, res in outcomes["claims"]} <= {"ok", "held"}
        holders = set(_ids_holders(snap).values())
        assert holders == {winners[0]}


def test_concurrent_finalize_and_release_finalize_never_sees_holders(store):
    # alice 持有整组；release 与 finalize(预期 revision=2) 并发：
    #  - release 先 → 无持有者但 revision=3，结稿版本过期 409（不会误成功）
    #  - finalize 先 → 持有者仍在，409；随后 release 成功
    cid = store.create(TEXT, [ann("a", 0, 4), ann("b", 3, 7)])["id"]
    store.claim(cid, "a", "alice")  # revision=2
    results = {"release": None, "finalize": None}
    barrier = threading.Barrier(2)

    def do_release():
        barrier.wait()
        try:
            store.release(cid, "b", "alice")
            results["release"] = "ok"
        except Exception as exc:  # noqa: BLE001 - 记录类型供断言
            results["release"] = type(exc).__name__

    def do_finalize():
        barrier.wait()
        try:
            store.finalize(cid, 2)
            results["finalize"] = "ok"
        except Exception as exc:  # noqa: BLE001
            results["finalize"] = type(exc).__name__

    t1 = threading.Thread(target=do_release)
    t2 = threading.Thread(target=do_finalize)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    snap = store.get(cid)
    assert results["release"] == "ok", results
    assert results["finalize"] in ("RevisionConflictError", "HoldersExistError")
    assert results["finalize"] != "ok"
    # 两种次序下都不可能已结稿
    assert snap["status"] == "open"


def test_concurrent_finalize_against_modify(store):
    # alice 持有 a；modify（revision 2→3）与 finalize(预期 revision=2) 并发：
    # modify 先 → finalize 版本过期 409；finalize 先（持有者存在）本就 409。
    cid = store.create(TEXT, [ann("a", 0, 2), ann("b", 12, 15)])["id"]
    store.claim(cid, "a", "alice")  # revision=2
    results = {"modify": None, "finalize": None}
    barrier = threading.Barrier(2)

    def do_modify():
        barrier.wait()
        try:
            store.modify(cid, "a", "alice", 0, 1, "新")
            results["modify"] = "ok"
        except Exception as exc:  # noqa: BLE001
            results["modify"] = type(exc).__name__

    def do_finalize():
        barrier.wait()
        try:
            store.finalize(cid, 2)
            results["finalize"] = "ok"
        except Exception as exc:  # noqa: BLE001
            results["finalize"] = type(exc).__name__

    t1 = threading.Thread(target=do_modify)
    t2 = threading.Thread(target=do_finalize)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    snap = store.get(cid)
    assert results["finalize"] != "ok"
    assert snap["status"] == "open"  # 有持有者，结稿本来就不可能成功
    assert results["modify"] == "ok"
    a = {x["id"]: x for x in snap["annotations"]}["a"]
    assert (a["end"], a["label"]) == (1, "新")
