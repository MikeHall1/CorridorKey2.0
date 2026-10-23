"""Console-capable frozen entry; worker pipes require real standard streams."""
import multiprocessing

from corridorkey.app_process import main

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
