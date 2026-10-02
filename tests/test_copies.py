"""可领取审阅工作副本：纯内存核心测试（分组 / 领取 / 释放 / 修改 / 结稿）。"""

import pytest

from app.copies import (
    AnnotationNotFound,
    CopyFinalizedError,
    CopyState,
    CopyValidationError,
    GroupHeldError,
    HoldersExistError,
    ModifyConflictError,
    NotHeldError,
    RevisionConflictError,
    check_interval,
    check_revision,
    connected_groups,
)

TEXT = "x" * 20


def ann(id, start, end, label="L"):
    return {"id": id, "start": start, "end": end, "label": label}


def state(anns):
    return CopyState("c1", TEXT, anns)


def holders_of(snap):
    return {a["id"]: a["holder"] for a in snap["annotations"]}


# ---------- 分组 ----------

def test_groups_positive_overlap_edges():
    # a[0,3) 与 b[2,5) 正长度重叠连边；b 与 c[5,7) 相接不连边
    g = connected_groups([ann("a", 0, 3), ann("b", 2, 5), ann("c", 5, 7)])
    assert g == [["a", "b"], ["c"]]


def test_groups_transitive_chain_is_one_component():
    # a[0,4) b[3,7) c[6,10) 传递连通整组
    g = connected_groups([ann("a", 0, 4), ann("b", 3, 7), ann("c", 6, 10)])
    assert g == [["a", "b", "c"]]


def test_groups_touching_does_not_connect():
    g = connected_groups([ann("a", 0, 4), ann("b", 4, 8)])
    assert g == [["a"], ["b"]]


def test_groups_nested_connects():
    g = connected_groups([ann("outer", 0, 10), ann("inner", 2, 4)])
    assert g == [["outer", "inner"]]


def test_groups_sorted_by_earliest_member_and_members_stable_sorted():
    # z 与第二簇重叠，组序仍按最早 start
    g = connected_groups([
        ann("z", 9, 12),
        ann("a", 0, 2),
        ann("m", 10, 11),  # 与 z 嵌套
        ann("b", 20, 22),
    ])
    assert g == [["a"], ["z", "m"], ["b"]]


def test_groups_empty():
    assert connected_groups([]) == []


# ---------- 领取 ----------

def test_initial_snapshot_unclaimed():
    snap = state([ann("a", 0, 3), ann("b", 2, 5)]).snapshot()
    assert snap["revision"] == 1
    assert holders_of(snap) == {"a": None, "b": None}
    assert snap["groups"] == [{"annotation_ids": ["a", "b"], "holder": None}]


def test_claim_takes_whole_transitive_group():
    s = state([ann("a", 0, 4), ann("b", 3, 7), ann("c", 6, 10), ann("d", 15, 18)])
    s.claim("a", "alice")
    snap = s.snapshot()
    assert holders_of(snap) == {"a": "alice", "b": "alice", "c": "alice",
                                "d": None}
    # 从组内任一批注 id 领取都是整组
    assert snap["groups"][0] == {"annotation_ids": ["a", "b", "c"],
                                 "holder": "alice"}


def test_claim_unconnected_not_taken():
    s = state([ann("a", 0, 3), ann("b", 5, 8)])
    s.claim("a", "alice")
    assert holders_of(s.snapshot()) == {"a": "alice", "b": None}


def test_claim_bumped_revision_then_idempotent():
    s = state([ann("a", 0, 3)])
    assert s.claim("a", "alice") is True
    assert s.revision == 2
    assert s.claim("a", "alice") is False  # 本人重复领取：成功但无新版本
    assert s.revision == 2


def test_claim_other_holders_group_conflict_unchanged():
    s = state([ann("a", 0, 4), ann("b", 3, 7)])
    s.claim("a", "alice")
    with pytest.raises(GroupHeldError) as exc:
        s.claim("b", "bob")
    assert exc.value.holder == "alice"
    assert set(exc.value.group) == {"a", "b"}
    assert holders_of(s.snapshot()) == {"a": "alice", "b": "alice"}
    assert s.revision == 2


def test_claim_unknown_annotation():
    with pytest.raises(AnnotationNotFound):
        state([]).claim("nope", "alice")


def test_claim_reviewer_must_be_nonempty_string():
    s = state([ann("a", 0, 3)])
    for bad in ("", None, 3, ["x"]):
        with pytest.raises(CopyValidationError):
            s.claim("a", bad)


# ---------- 释放 ----------

def test_release_only_holder_can_release_whole_group():
    s = state([ann("a", 0, 4), ann("b", 3, 7), ann("c", 12, 15)])
    s.claim("a", "alice")
    with pytest.raises(GroupHeldError):
        s.release("b", "bob")  # 他人不能释放
    s.release("b", "alice")   # 从组内任一条释放整组
    assert holders_of(s.snapshot()) == {"a": None, "b": None, "c": None}
    assert s.revision == 3


def test_release_unheld_conflict():
    s = state([ann("a", 0, 3)])
    with pytest.raises(NotHeldError):
        s.release("a", "alice")
    assert s.revision == 1


# ---------- 修改 ----------

def test_modify_requires_holder():
    s = state([ann("a", 0, 3)])
    with pytest.raises(NotHeldError):
        s.modify("a", "alice", 0, 2, "L")
    s.claim("a", "alice")
    s.modify("a", "alice", 1, 2, "新label")
    a = s.snapshot()["annotations"][0]
    assert (a["start"], a["end"], a["label"]) == (1, 2, "新label")


def test_modify_label_only_does_not_touch_geometry():
    s = state([ann("a", 0, 4), ann("b", 3, 7)])
    s.claim("a", "alice")
    s.modify("b", "alice", 3, 7, "X")
    snap = s.snapshot()
    assert holders_of(snap) == {"a": "alice", "b": "alice"}
    assert snap["groups"][0]["annotation_ids"] == ["a", "b"]


def test_modify_invalid_range_or_label_is_422_and_unchanged():
    s = state([ann("a", 0, 3)])
    s.claim("a", "alice")
    for kwargs in (dict(start=3, end=3, label="L"),   # 零长
                   dict(start=-1, end=2, label="L"),
                   dict(start=0, end=99, label="L"),  # 超文本
                   dict(start=0, end=2, label=""),
                   dict(start=0, end=2, label=None)):
        with pytest.raises(CopyValidationError):
            s.modify("a", "alice", **kwargs)
    a = s.snapshot()["annotations"][0]
    assert (a["start"], a["end"], a["label"]) == (0, 3, "L")
    assert s.revision == 2  # 仅领取产生过一次版本


def test_modify_merges_unclaimed_annotations_to_holder():
    # a（alice 持有）扩大后与未领取的 b 相接式重叠 → b 并入归 alice
    s = state([ann("a", 0, 2), ann("b", 4, 7)])
    s.claim("a", "alice")
    s.modify("a", "alice", 0, 5, "L")  # 与 b[4,7) 正长度重叠
    snap = s.snapshot()
    assert [g["annotation_ids"] for g in snap["groups"]] == [["a", "b"]]
    assert holders_of(snap) == {"a": "alice", "b": "alice"}


def test_modify_touching_other_holder_conflicts_and_nothing_changes():
    # a（alice）扩大后只与 b（bob）相接（端点相等）→ 不连边，不算触及
    s = state([ann("a", 0, 2), ann("b", 2, 5)])
    s.claim("a", "alice")
    s.claim("b", "bob")
    s.modify("a", "alice", 0, 2, "X")  # 仅改 label，范围不变，相接不连边
    assert holders_of(s.snapshot()) == {"a": "alice", "b": "bob"}


def test_modify_overlapping_other_holder_conflicts_atomically():
    # a（alice）扩大后与 b（bob）正长度重叠 → 409，范围与持有权均不变
    s = state([ann("a", 0, 2), ann("b", 4, 7)])
    s.claim("a", "alice")
    s.claim("b", "bob")
    with pytest.raises(ModifyConflictError) as exc:
        s.modify("a", "alice", 0, 5, "L")
    assert exc.value.foreign == {"b": "bob"}
    a, b = s.snapshot()["annotations"]
    assert (a["start"], a["end"]) == (0, 2)
    assert holders_of(s.snapshot()) == {"a": "alice", "b": "bob"}
    assert s.revision == 3  # 两次领取；失败修改不增版本


def test_modify_split_group_keeps_original_holder():
    # a、b 同属 alice 的组；把 a 移到远处拆开，两碎片都保留 alice
    s = state([ann("a", 0, 4), ann("b", 3, 7)])
    s.claim("a", "alice")
    s.modify("a", "alice", 15, 18, "L")
    snap = s.snapshot()
    assert [g["annotation_ids"] for g in snap["groups"]] == [["b"], ["a"]]
    assert all(g["holder"] == "alice" for g in snap["groups"])
    assert holders_of(snap) == {"a": "alice", "b": "alice"}


def test_modify_split_releases_other_holders_pieces_not_touched():
    # 同组 a(alice)+b(alice)，远处还有 bob 持有 c；a 移到 c 旁但仅相接 →
    # 不合并，bob 的 c 完全不受影响
    s = state([ann("a", 0, 4), ann("b", 3, 7), ann("c", 18, 20)])
    s.claim("a", "alice")
    s.claim("c", "bob")
    s.modify("a", "alice", 15, 18, "L")  # 与 c[18,20) 相接
    assert holders_of(s.snapshot()) == {"a": "alice", "b": "alice", "c": "bob"}


def test_modify_unknown_annotation():
    s = state([])
    with pytest.raises(AnnotationNotFound):
        s.modify("x", "alice", 0, 1, "L")


def test_check_interval_helper():
    check_interval(0, 1, 5)
    check_interval(4, 5, 5)
    for bad in ((0, 0, 5), (-1, 2, 5), (3, 6, 5), (4, 3, 5)):
        with pytest.raises(CopyValidationError):
            check_interval(*bad)


# ---------- 结稿（finalize） ----------

def test_check_revision_helper():
    check_revision(1)
    check_revision(10 ** 9)
    for bad in (0, -1, True, False, None, "1", 1.0, [], {}):
        with pytest.raises(CopyValidationError):
            check_revision(bad)


def test_initial_status_is_open():
    snap = state([ann("a", 0, 3)]).snapshot()
    assert snap["status"] == "open"


def test_finalize_unheld_succeeds_bumps_only_status_and_revision():
    s = state([ann("a", 0, 4), ann("b", 3, 7), ann("c", 15, 18)])
    before = s.snapshot()
    s.finalize(1)
    snap = s.snapshot()
    assert snap["status"] == "finalized"
    assert snap["revision"] == 2
    # 文本、批注内容、持有权与排序均不变
    assert snap["text"] == before["text"]
    assert [(a["id"], a["start"], a["end"], a["label"], a["holder"])
            for a in snap["annotations"]] == \
        [(a["id"], a["start"], a["end"], a["label"], a["holder"])
         for a in before["annotations"]]
    assert snap["groups"] == before["groups"]


def test_finalize_stale_revision_conflicts_and_changes_nothing():
    s = state([ann("a", 0, 3)])
    with pytest.raises(RevisionConflictError) as exc:
        s.finalize(2)  # 当前 revision=1
    assert exc.value.expected == 2 and exc.value.current == 1
    snap = s.snapshot()
    assert snap["status"] == "open" and snap["revision"] == 1


def test_finalize_with_holders_conflicts_and_changes_nothing():
    s = state([ann("a", 0, 4), ann("b", 3, 7), ann("c", 15, 18)])
    s.claim("c", "bob")  # c 独立成组被持有，revision 1→2
    with pytest.raises(HoldersExistError) as exc:
        s.finalize(2)
    assert exc.value.holders == ["bob"]
    # 版本过期先于持有者检查
    with pytest.raises(RevisionConflictError):
        s.finalize(1)
    snap = s.snapshot()
    assert snap["status"] == "open" and snap["revision"] == 2  # 仅领取增过版本
    assert holders_of(snap) == {"a": None, "b": None, "c": "bob"}


def test_finalize_already_finalized_conflicts():
    s = state([ann("a", 0, 3)])
    s.finalize(1)
    with pytest.raises(CopyFinalizedError):
        s.finalize(2)  # 即使版本正确，已结稿仍 409；不提供重开
    assert s.snapshot()["revision"] == 2


def test_finalize_bad_revision_is_validation_error_before_other_checks():
    s = state([ann("a", 0, 3)])
    for bad in (0, -1, True, False, None, "1", 1.0):
        with pytest.raises(CopyValidationError):
            s.finalize(bad)
    assert s.snapshot()["status"] == "open"
    # 已结稿后非法版本号仍是 422（非法输入优先）
    s.finalize(1)
    with pytest.raises(CopyValidationError):
        s.finalize(0)


def test_claim_release_modify_after_finalized_all_conflict_but_404_422_kept():
    s = state([ann("a", 0, 3), ann("b", 5, 8)])
    s.finalize(1)
    # 有效领取 / 释放 / 修改请求 → 409
    with pytest.raises(CopyFinalizedError):
        s.claim("a", "alice")
    with pytest.raises(CopyFinalizedError):
        s.release("a", "alice")
    with pytest.raises(CopyFinalizedError):
        s.modify("a", "alice", 0, 1, "X")
    # 404 含义保留：批注不存在仍报 AnnotationNotFound（不是结稿错误）
    with pytest.raises(AnnotationNotFound):
        s.claim("nope", "alice")
    # 422 含义保留：非法 reviewer / 区间仍报校验错误（优先于结稿门禁）
    with pytest.raises(CopyValidationError):
        s.claim("a", "")
    with pytest.raises(CopyValidationError):
        s.modify("a", "alice", 0, 0, "X")
    snap = s.snapshot()
    assert snap["revision"] == 2 and snap["status"] == "finalized"
    assert holders_of(snap) == {"a": None, "b": None}


def test_finalized_idempotent_claim_still_conflicts():
    # 未结稿时本人重复领取幂等；结稿后一律 409（无幂等例外）
    s = state([ann("a", 0, 3)])
    s.finalize(1)
    with pytest.raises(CopyFinalizedError):
        s.claim("a", "alice")
    assert s.snapshot()["revision"] == 2
