"""Double-click starts the GUI; subcommands provide developer automation."""
import argparse,json,sys
from pathlib import Path

def main() -> int:
    import multiprocessing
    multiprocessing.freeze_support()
    from hukou.offline import install_guard
    install_guard()
    from hukou.runtime import configure_runtime,ThreadSettings
    configure_runtime(ThreadSettings())
    if sys.stdout: sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr: sys.stderr.reconfigure(encoding='utf-8')
    if len(sys.argv)==1:
        from hukou.gui import launch
        return launch()
    parser=argparse.ArgumentParser(description='户口资料批量整理工具（离线）')
    subs=parser.add_subparsers(dest='command',required=True)
    for name in ('parse','batch'):
        p=subs.add_parser(name); p.add_argument('input',type=Path); p.add_argument('--output',type=Path)
        p.add_argument('--dpi',type=int,choices=(200,250,300),default=250)
        if name=='batch':
            p.add_argument('--naming',choices=('name','name_number'),default='name')
            p.add_argument('--force',action='store_true'); p.add_argument('--recursive',action='store_true')
            p.add_argument('--mode',choices=('auto','low','standard','high','custom'),default='auto')
            p.add_argument('--workers',type=int,choices=range(1,7))
    p=subs.add_parser('gui'); p.add_argument('--smoke',type=Path)
    args=parser.parse_args()
    if args.command=='gui':
        from hukou.gui import launch
        return launch(args.smoke)
    from hukou.pipeline import Pipeline
    from hukou.config import Settings
    from hukou.storage import atomic_json
    try:
        if args.command=='parse':
            pipe=Pipeline(Settings(dpi=args.dpi))
            h=pipe.parse(args.input)
            target=args.output or args.input.parent/'输出'/f'{args.input.stem}.json'
            if target.suffix.lower()!='.json' or target.resolve()==args.input.resolve():
                raise ValueError('单户输出必须是独立的JSON文件，不能覆盖输入PDF')
            atomic_json(target,h.to_dict())
            result=dict(status=h.status,pages=h.page_count,members=len(h.members),issues=len(h.issues),output=str(target))
        else:
            from hukou.batch import batch
            result=batch(args.input,args.output,args.naming,args.force,recursive=args.recursive,
                         mode=args.mode,workers=args.workers,settings=Settings(dpi=args.dpi))
        if sys.stdout: print(json.dumps(result,ensure_ascii=False))
        return 1 if result.get('error') or result.get('status')=='error' else 0
    except Exception as exc:
        if sys.stderr: print('处理失败：'+type(exc).__name__,file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
