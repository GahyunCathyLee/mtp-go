import warnings
import os.path
from config import parse_config
from base_mdn import *
from datamodule import *
from models.gru_gnn import *
from lightning.pytorch import Trainer, seed_everything

torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

args = parse_config()


def main(encoder, decoder):
    ckpt_path = f"{args.ckpt_dir}/{args.config_name}/best.ckpt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    use_cuda = args.use_cuda and torch.cuda.is_available()
    devices, accelerator = (-1, "auto") if use_cuda else (1, "cpu")

    model = LitEncoderDecoder.load_from_checkpoint(ckpt_path, encoder=encoder, decoder=decoder, args=args)
    datamodule = LitDataModule(args)

    trainer = Trainer(accelerator=accelerator, devices=devices)
    results = trainer.test(model, datamodule=datamodule, verbose=True)[0]

    if not args.dry_run:
        from json import dumps
        json_object = dumps(results, indent=4)
        out_path = f"{args.ckpt_dir}/{args.config_name}/results.json"
        with open(out_path, "w") as f:
            f.write(json_object)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    seed_everything(args.seed, workers=True)
    if args.dataset in ('highD', 'highD-imp'):
        input_len = 2
        v_types = 2
    elif args.dataset == 'rounD':
        input_len = 3
        v_types = 7
    else:
        input_len = 3
        v_types = 4

    if args.dataset == 'highD-imp':
        n_features = 7 if getattr(args, 'use_importance', True) else 6
    else:
        n_features = 9
    static_f_dim = v_types * int(args.n_ode_static)

    dt = 2e-1
    max_l = int(input_len * (1 / dt)) + 1

    if args.motion_model == '1Xint':
        m_model = SingleIntegrator(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
    elif args.motion_model == '2Xint':
        m_model = DoubleIntegrator(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
    elif args.motion_model == '3Xint':
        m_model = TripleIntegrator(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
    elif args.motion_model == 'singletrack':
        m_model = KinematicSingleTrack(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
        if args.init_static:
            static_f_dim = 2
    elif args.motion_model == 'unicycle':
        m_model = Unicycle(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
    elif args.motion_model == 'curvature':
        m_model = Curvature(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
    elif args.motion_model == 'curvilinear':
        m_model = CurviLinear(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures, u1_lim=args.u1_lim)
    elif args.motion_model == 'neuralode':
        m_model = FirstOrderNeuralODE(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures,
                                      static_f_dim=static_f_dim, n_hidden=args.n_ode_hidden,
                                      n_layers=args.n_ode_layers)
    else:
        m_model = SecondOrderNeuralODE(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures,
                                       static_f_dim=static_f_dim, n_hidden=args.n_ode_hidden,
                                       n_layers=args.n_ode_layers)

    print(f'----------------------------------------------------')
    print(f'\nConfig : {args.config_name}  ({args.config_path})')
    print(f'Ckpt   : {args.ckpt_dir}/{args.config_name}/best.ckpt\n')
    print(f'----------------------------------------------------')

    encoder = GRUGNNEncoder(input_size=n_features,
                            hidden_size=args.hidden_size,
                            n_mixtures=m_model.mixtures,
                            n_layers=args.n_gnn_layers,
                            gnn_layer=args.gnn_layer,
                            n_heads=args.n_heads,
                            static_f_dim=static_f_dim,
                            init_static=args.init_static,
                            use_edge_features=args.use_edge_features)
    decoder = GRUGNNDecoder(m_model,
                            hidden_size=encoder.hidden_size,
                            max_length=max_l,
                            n_layers=args.n_gnn_layers,
                            n_heads=args.n_heads,
                            static_f_dim=static_f_dim,
                            gnn_layer=args.gnn_layer,
                            init_static=args.init_static)

    main(encoder, decoder)
