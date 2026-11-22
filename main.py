from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'source'))
from source.main import main
if __name__=='__main__': raise SystemExit(main())
