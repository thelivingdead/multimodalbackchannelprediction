"""IoU, matching, labels, leakage."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotations import parse_label
from src.events import Event, greedy_match, iou, match_pairs
from src.plotting import FigureLog, save_publication_figure


def test_match_pairs_and_greedy() -> None:
    a = Event("v", 1.0, 2.0)
    b = Event("v", 1.2, 2.2)
    extra = Event("v", 8.0, 8.4)
    assert iou(a, b) > 0.5
    tps, fps, fns = match_pairs([a, extra], [b], 0.3)
    assert len(tps) == 1
    assert fps == [extra]
    assert fns == []
    tp, fp, fn = greedy_match([a, extra], [b], 0.3)
    assert (tp, fp, fn) == (1, 1, 0)


def test_clock_mmss() -> None:
    from src.utils import format_mmss, parse_clock

    assert format_mmss(55) == "0:55"
    assert format_mmss(115) == "1:55"
    assert format_mmss(701.4) == "11:41"
    assert parse_clock("0:55") == 55
    assert parse_clock("11:41") == 701
    assert parse_clock("1:02") == 62


def test_parse_label() -> None:
    assert parse_label(1) == 1
    assert parse_label("0") == 0
    assert parse_label("clear") == 1
    assert parse_label("unclear") == 0
    assert parse_label("") is None


def test_split_files_no_overlap() -> None:
    root = Path(__file__).resolve().parents[1]
    dev = root / "data" / "splits" / "gold_dev.txt"
    tes = root / "data" / "splits" / "gold_test.txt"
    if not dev.exists() or not tes.exists():
        return
    d = {x.strip() for x in dev.read_text().splitlines() if x.strip()}
    t = {x.strip() for x in tes.read_text().splitlines() if x.strip()}
    assert d.isdisjoint(t)


def test_videomae_shake_path_isolation() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import check_split_leakage as gate

    nod = gate.assert_videomae_task_isolation(
        gold_csv=root / "data" / "gold_annotations.csv",
        label_col="label",
        pseudo_labels=root / "results" / "pseudo_labels.csv",
        out_dir=root / "results" / "videomae_finetuned",
    )
    assert nod == "head_nod"
    nod_head = gate.assert_videomae_task_isolation(
        gold_csv=root / "data" / "gold_annotations.csv",
        label_col="label",
        pseudo_labels=root / "results" / "pseudo_labels.csv",
        out_dir=root / "results" / "videomae_frozen_head",
        model_pt=root / "models" / "videomae_head.pt",
    )
    assert nod_head == "head_nod"

    shake = gate.assert_videomae_task_isolation(
        gold_csv=root / "data" / "gold" / "shake_annotation_sheet.csv",
        label_col="shake_label",
        pseudo_labels=root / "results" / "shake" / "pseudo_labels.csv",
        out_dir=root / "results" / "shake" / "videomae_finetuned",
    )
    assert shake == "head_shake"
    shake_head = gate.assert_videomae_task_isolation(
        gold_csv=root / "data" / "gold" / "shake_annotation_sheet.csv",
        label_col="shake_label",
        pseudo_labels=root / "results" / "shake" / "pseudo_labels.csv",
        out_dir=root / "results" / "shake" / "videomae_frozen_head",
        model_pt=root / "results" / "shake" / "videomae_frozen_head" / "best_model.pt",
    )
    assert shake_head == "head_shake"

    try:
        gate.assert_videomae_task_isolation(
            gold_csv=root / "data" / "gold_annotations.csv",
            label_col="label",
            pseudo_labels=root / "results" / "shake" / "pseudo_labels.csv",
            out_dir=root / "results" / "shake" / "videomae_finetuned",
        )
    except SystemExit as exc:
        msg = str(exc)
        assert "shake_label" in msg
        assert "videomae_finetuned" in msg or "gold_annotations" in msg
    else:
        raise AssertionError("mixed nod gold + shake out-dir must abort")

    try:
        gate.assert_videomae_task_isolation(
            gold_csv=root / "data" / "gold" / "shake_annotation_sheet.csv",
            label_col="shake_label",
            pseudo_labels=root / "results" / "shake" / "pseudo_labels.csv",
            out_dir=root / "results" / "videomae_finetuned",
        )
    except SystemExit as exc:
        assert "results/shake" in str(exc) or "videomae_finetuned" in str(exc)
    else:
        raise AssertionError("shake labels must not write nod out-dir")

    try:
        gate.assert_videomae_task_isolation(
            gold_csv=root / "data" / "gold" / "shake_annotation_sheet.csv",
            label_col="shake_label",
            pseudo_labels=root / "results" / "shake" / "pseudo_labels.csv",
            out_dir=root / "results" / "videomae_frozen_head",
            model_pt=root / "models" / "videomae_head.pt",
        )
    except SystemExit as exc:
        msg = str(exc)
        assert "videomae_frozen_head" in msg or "videomae_head.pt" in msg
        assert "results/shake" in msg or "videomae_head.pt" in msg
    else:
        raise AssertionError("shake must not write nod frozen-head artefacts")

    try:
        gate.assert_videomae_task_isolation(
            gold_csv=root / "data" / "gold_annotations.csv",
            label_col="label",
            pseudo_labels=root / "results" / "pseudo_labels.csv",
            out_dir=root / "results" / "shake" / "videomae_frozen_head",
        )
    except SystemExit as exc:
        assert "results/shake" in str(exc) or "shake_label" in str(exc)
    else:
        raise AssertionError("nod run must not write under results/shake/")

    try:
        gate.assert_joint_videomae_paths(
            gold_csv=root / "data" / "gold" / "shake_annotation_sheet.csv",
            nod_pseudo=root / "results" / "pseudo_labels.csv",
            shake_pseudo=root / "results" / "shake" / "pseudo_labels.csv",
            out_dir=root / "results" / "joint" / "videomae_finetuned",
        )
    except SystemExit as exc:
        msg = str(exc).lower()
        assert "locked" in msg or "joint" in msg
    else:
        raise AssertionError(
            "locked results/joint/videomae_finetuned must be refused"
        )
    joint = gate.assert_joint_videomae_paths(
        gold_csv=root / "data" / "gold" / "shake_annotation_sheet.csv",
        nod_pseudo=root / "results" / "pseudo_labels.csv",
        shake_pseudo=root / "results" / "shake" / "pseudo_labels.csv",
        out_dir=root / "results" / "joint" / "dev_unscored_run",
    )
    assert joint == "joint_nod_shake"
    for blocked in gate.LOCKED_OUT_DIRS:
        try:
            gate.assert_unlocked_out_dir(blocked)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"locked dir must be refused: {blocked}")
    try:
        gate.assert_joint_videomae_paths(
            gold_csv=root / "data" / "gold" / "shake_annotation_sheet.csv",
            nod_pseudo=root / "results" / "pseudo_labels.csv",
            shake_pseudo=root / "results" / "shake" / "pseudo_labels.csv",
            out_dir=root / "results" / "shake" / "videomae_finetuned",
        )
    except SystemExit as exc:
        assert "joint" in str(exc).lower() or "locked" in str(exc).lower()
    else:
        raise AssertionError("joint run must not write locked shake VideoMAE dir")
    try:
        gate.assert_unlocked_out_dir(
            root / "results" / "shake" / "videomae_finetuned"
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("locked shake finetune dir must be refused")


def test_shake_balance_subsample_75_5() -> None:
    import numpy as np

    from src.balance import balance_indices, boosted_pos_weight

    y = np.array([1] * 75 + [0] * 5)
    idx = balance_indices(y, mode="subsample", ratio=1.0, seed=42)
    yy = y[idx]
    assert int((yy == 1).sum()) == 5
    assert int((yy == 0).sum()) == 5
    ov = balance_indices(y, mode="oversample", ratio=1.0, seed=42)
    yo = y[ov]
    assert int((yo == 1).sum()) == 75
    assert int((yo == 0).sum()) == 75
    # majority is positive: boost should shrink pos_weight
    assert boosted_pos_weight(75, 5, boost=1.0) == 5 / 75
    assert boosted_pos_weight(75, 5, boost=5.0) == (5 / 75) / 5.0


def test_majority_always_positive_from_gold() -> None:
    import pandas as pd

    from src.clip_metrics import always_predict

    root = Path(__file__).resolve().parents[1]
    shake = pd.read_csv(root / "data" / "gold" / "shake_annotation_sheet.csv")
    y = shake.loc[
        shake["split"].astype(str).str.upper() == "TEST", "shake_label"
    ].astype(int).to_numpy()
    m = always_predict(y, 1)
    assert len(y) == 15
    assert int((y == 1).sum()) == 7
    assert m["tp"] == 7 and m["fp"] == 8 and m["tn"] == 0 and m["fn"] == 0
    assert abs(m["f1"] - (2 * (7 / 15) * 1.0) / ((7 / 15) + 1.0)) < 1e-9

    nod = pd.read_csv(root / "data" / "gold_annotations.csv")
    yn = nod.loc[
        nod["split"].astype(str).str.upper() == "TEST", "label"
    ].astype(int).to_numpy()
    mn = always_predict(yn, 1)
    assert int((yn == 1).sum()) == 10
    assert abs(mn["f1"] - 0.8) < 1e-9


def test_shake_collapse_diagnostics() -> None:
    import numpy as np

    from src.clip_metrics import collapse_diagnostics

    always1 = collapse_diagnostics(np.ones(15, dtype=int), tn=0)
    assert always1["collapse"] is True
    assert always1["predicted_positive_rate"] == 1.0
    mixed = collapse_diagnostics(np.array([1] * 10 + [0] * 5), tn=2)
    assert mixed["predicted_positive_rate"] == 10 / 15
    # 10/15 ≈ 0.667, not collapse by rate; TN=2 so not collapse
    assert mixed["collapse"] is False
    high_rate = collapse_diagnostics(np.array([1] * 13 + [0] * 2), tn=2)
    assert high_rate["predicted_positive_rate"] > 0.85
    assert high_rate["collapse"] is True


def test_choose_dev_threshold_dev_only() -> None:
    import ast
    import inspect

    import numpy as np

    from src.clip_metrics import choose_dev_threshold

    sig = inspect.signature(choose_dev_threshold)
    assert list(sig.parameters) == ["y_true", "prob", "criterion"]

    y = np.array([0, 0, 1, 1])
    thr, m = choose_dev_threshold(y, np.array([0.1, 0.1, 0.9, 0.9]), criterion="f1")
    assert m["f1"] == 1.0
    assert m["tn"] == 2
    assert 0.2 <= thr <= 0.8

    _, m_all = choose_dev_threshold(y, np.full(4, 0.9), criterion="f1")
    assert m_all["tn"] == 0
    assert m_all["recall"] == 1.0

    pose_src = Path(__file__).resolve().parents[1] / "src" / "pose_cnn.py"
    tree = ast.parse(pose_src.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_matrix"
    )
    assert "seq_len" in {a.arg for a in fn.args.args}


def test_dev_search_dir_is_unlocked() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import check_split_leakage as gate

    gate.assert_unlocked_out_dir(
        root / "results" / "shake" / "dev_balanced" / "cnn_A_40_40"
    )
    gate.assert_unlocked_out_dir(
        root / "results" / "shake" / "dev_search" / "cnn_MAX_all"
    )
    try:
        gate.assert_unlocked_out_dir(
            root / "results" / "shake" / "videomae_frozen_head_balanced"
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("TEST-scored balanced frozen dir must stay locked")


def test_shake_videomae_leakage_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import check_split_leakage as gate

    gate.run(
        gold_csv=root / "data" / "gold" / "shake_annotation_sheet.csv",
        pseudo_labels=root / "results" / "shake" / "pseudo_labels.csv",
        labelled_train_only=True,
    )


def _ast_main_arg_names(path: Path) -> set[str]:
    import ast

    tree = ast.parse(path.read_text())
    mains = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert mains, f"no main() in {path}"
    fn = mains[0]
    names = {arg.arg for arg in fn.args.args}
    names.update(arg.arg for arg in fn.args.kwonlyargs)
    return names


def test_shake_dev_wrappers_use_cli_not_kwargs() -> None:
    """DEV wrappers must not pass kwargs the nod trainer may reject."""
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    head = (scripts / "train_videomae_shake_head_dev.py").read_text()
    ft = (scripts / "finetune_videomae_shake_dev.py").read_text()
    cnn = (scripts / "train_shake_cnn_dev.py").read_text()
    sh = (scripts / "run_shake_dev_search.sh").read_text()
    assert "score_test=" not in head
    assert "score_test=" not in ft
    assert "select_dev=" not in head.split("train_head")[-1]
    assert "--dev-only" in head
    assert "--dev-only" in ft
    assert "--dev-only" in cnn
    assert "scripts/run_shake_dev_search.py" not in sh
    assert "train_shake_cnn_dev.py" in sh
    assert "train_videomae_shake_head_dev.py" in sh
    assert "finetune_videomae_shake_dev.py" in sh
    assert "compare_shake_dev_search.py" in sh


def test_videomae_and_cnn_main_accept_dev_only() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    for rel, needed in (
        (
            "train_videomae_head.py",
            {"argv", "gold_csv", "dev_only", "score_test", "select_dev", "seed"},
        ),
        (
            "finetune_videomae.py",
            {"argv", "gold_csv", "dev_only", "score_test", "select_dev", "seed"},
        ),
        ("train_shake_cnn.py", {"argv"}),
    ):
        names = _ast_main_arg_names(scripts / rel)
        missing = needed - names
        assert not missing, f"{rel} main() missing {sorted(missing)}"


def test_save_publication_figure_writes_png_and_jpg(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    stem = tmp_path / "unit_test_fig"
    log = FigureLog()
    ok = save_publication_figure(fig, stem, log, source="test", force=True)
    assert ok
    assert stem.with_suffix(".png").exists()
    assert stem.with_suffix(".jpg").exists()
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [1, 0])
    ok2 = save_publication_figure(fig2, stem, log, source="test", force=False)
    assert ok2 is False
