"""Обучение и выгрузка модели - только по команде.

    python -m game.train trial --minutes 10                  # замер скорости и прогноз времени
    python -m game.train train --run runs/main --games 1000000 [--hours 6] [--resume]
    python -m game.train export --run runs/main --snapshot latest --out game/models/net.npz

Потоки библиотек ограничены одним, процесс идёт с пониженным приоритетом: ПК общий.
"""
import argparse
import os
import sys

# До импорта numpy: иначе ограничение потоков может не подействовать. Список переменных -
# тот же, что td.THREAD_VARS (td здесь ещё нельзя импортировать: он тянет numpy).
THREADS = os.environ.get("AI2048_TRAIN_THREADS", "1")
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = THREADS


def _trial(args):
    import time

    from game.ai.ntuple import NTupleNet
    from game.train import td

    run = args.run
    tr = td.Trainer(NTupleNet(), batch=args.batch, alpha=args.alpha, seed=args.seed, run_dir=run,
                    log_every=args.log_every, snapshot_every=10 ** 9, echo=True)
    started = time.time()
    tr.run(games=10 ** 9, seconds=args.minutes * 60)
    took = time.time() - started
    s = tr.stats()
    mps = tr.moves_done / took
    print()
    print(f"замер: {tr.games_done} партий за {took / 60:.1f} мин, {mps:.0f} ходов/с, "
          f"средний счёт последних партий {s['avg_score']:.0f}, ходов в партии {s['avg_moves']:.0f}, "
          f"2048 в {100 * s['reached_2048']:.1f}%")
    print("прогноз (ходов в партии у обученной сети ~ счёт/22; берётся --expected-moves):")
    for games in (100_000, 200_000, 500_000, 1_000_000):
        hours = games * args.expected_moves / mps / 3600
        print(f"  {games:>9} партий: ~{hours:.1f} ч")
    return 0


def _train(args):
    from game.ai.ntuple import NTupleNet
    from game.train import td

    if args.resume and os.path.exists(os.path.join(args.run, "state.json")):
        over = {}
        if args.alpha_set:
            over["alpha"] = args.alpha
        tr = td.Trainer.resume(args.run, **over)
        tr.echo = True
        print(f"продолжение: {tr.games_done} партий, {tr.seconds / 3600:.2f} ч, alpha {tr.alpha}")
    else:
        milestones = [int(m) for m in args.milestones.split(",") if m]
        tr = td.Trainer(NTupleNet(), batch=args.batch, alpha=args.alpha, seed=args.seed,
                        run_dir=args.run, log_every=args.log_every,
                        snapshot_every=args.snapshot_every, milestones=milestones, echo=True)
    seconds = args.hours * 3600 if args.hours else None
    stop_file = os.path.join(args.run, "STOP")

    def should_stop():
        return os.path.exists(stop_file)

    if args.decay_at and tr.games_done < args.decay_at:
        tr.run(games=min(args.decay_at, args.games), seconds=seconds, should_stop=should_stop)
        if seconds is not None:
            seconds = max(0.0, seconds - tr.seconds)
        if tr.games_done >= args.decay_at and not should_stop():
            tr.alpha = args.decay_alpha
            print(f"шаг обучения уменьшен до {tr.alpha} после {tr.games_done} партий", flush=True)
            tr.save()
    elif args.decay_at and tr.games_done >= args.decay_at and tr.alpha != args.decay_alpha:
        tr.alpha = args.decay_alpha
    if tr.games_done < args.games and not should_stop() and (seconds is None or seconds > 0):
        tr.run(games=args.games, seconds=seconds, should_stop=should_stop)
    s = tr.stats()
    print(f"итог: {s['games']} партий, {s['elapsed_s'] / 3600:.2f} ч, средний счёт {s['avg_score']:.0f}, "
          f"2048 в {100 * s['reached_2048']:.1f}%", flush=True)
    return 0


def _export(args):
    import json

    from game.ai.ntuple import NTupleNet

    with open(os.path.join(args.run, "state.json"), encoding="utf-8") as f:
        state = json.load(f)
    tuples = [tuple(t) for t in state["tuples"]]
    net = NTupleNet.load_raw(os.path.join(args.run, f"{args.snapshot}.npy"), tuples)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    net.export(args.out, meta={"snapshot": args.snapshot})
    print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.1f} МБ")
    return 0


def _passport(args):
    import json

    from game.ai import levels
    from game.train import passport

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(levels.__file__)), "..", "models",
                                   levels.PASSPORT)
    out = os.path.abspath(out)
    extra = {}
    if args.novice_games:
        extra["novice_snapshot_games"] = args.novice_games
    only = [x for x in args.only.split(",") if x] if args.only else None
    p = passport.build(out, args.run, args.games, only=only, extra=extra,
                       echo=lambda s: print(s, flush=True))
    print(json.dumps({k: {"avg": v["avg_score"], "2048": v["reached"]["2048"]}
                      for k, v in p["results"].items()}, ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m game.train", description="Обучение n-tuple сети 2048")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--batch", type=int, default=256)
        p.add_argument("--alpha", type=float, default=0.1)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--log-every", type=int, default=2000)

    t = sub.add_parser("trial", help="замер скорости и прогноз времени")
    common(t)
    t.add_argument("--minutes", type=float, default=10)
    t.add_argument("--run", default=os.path.join("runs", "trial"))
    t.add_argument("--expected-moves", type=float, default=4500)

    r = sub.add_parser("train", help="полное обучение")
    common(r)
    r.add_argument("--run", default=os.path.join("runs", "main"))
    r.add_argument("--games", type=int, default=1_000_000)
    r.add_argument("--hours", type=float, default=None, help="предел времени этого запуска")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--snapshot-every", type=int, default=20_000)
    r.add_argument("--milestones", default="500,1000,2000,5000,20000,100000")
    r.add_argument("--decay-at", type=int, default=0, help="после стольких партий уменьшить шаг")
    r.add_argument("--decay-alpha", type=float, default=0.01)

    e = sub.add_parser("export", help="сжатая модель для игры")
    e.add_argument("--run", default=os.path.join("runs", "main"))
    e.add_argument("--snapshot", default="latest")
    e.add_argument("--out", required=True)

    pp = sub.add_parser("passport", help="проверки силы: по 1000 партий на уровень")
    pp.add_argument("--run", default=None, help="папка обучения - сведения об обучении")
    pp.add_argument("--games", type=int, default=1000)
    pp.add_argument("--out", default=None)
    pp.add_argument("--only", default="", help="через запятую: random,corner,novice,medium,strong")
    pp.add_argument("--novice-games", type=int, default=0, help="после скольких партий снят ранний снимок")

    args = ap.parse_args(argv)
    args.alpha_set = any(a.startswith("--alpha") for a in (argv if argv is not None else sys.argv[1:]))
    from game.train.td import lower_priority
    lower_priority()
    return {"trial": _trial, "train": _train, "export": _export,
            "passport": _passport}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
