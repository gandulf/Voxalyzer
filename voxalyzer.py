# Compilation mode, standalone everywhere, except on macOS there app bundle
# nuitka-project-if: {OS} in ("Windows", "Linux", "FreeBSD"):
#    nuitka-project: --mode=onefile
#    nuitka-project: --windows-console-mode=force
# nuitka-project-else:
#    nuitka-project: --mode=onefile
#    nuitka-project: --macos-create-app-bundle
#
# nuitka-project: --include-data-dir={MAIN_DIRECTORY}/models=models
# nuitka-project: --mingw64
# nuitka-project: --output-dir=dist
import logging

from analyzer_onnx import analyze_files
from mp3 import clean_mp3, list_mp3s, update_mp3_results
from utils import AnalyzeResult

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
import os
import sys

import traceback

MODEL_VERSION = "2.0"

def handle_analyze_result(file_path: str, analysis: AnalyzeResult):
    logger.info(f"Analyzed {file_path}:")

    update_mp3_results(file_path, analysis)

    for key, item in analysis.items():
        logger.info(f"  {key}: {item}")

def main():
    args = [arg for arg in sys.argv if not arg.endswith(".exe") and not arg.endswith(".py")]

    port = 8000
    try:
        index = args.index("--port")
        args.pop(index)
        port = int(args.pop(index))
    except ValueError:
        pass

    if len(args) == 0:
        import server
        server.serve(port)
    else:
        force = False
        clean = False
        if "--force" in args:
            args.remove("--force")
            force = True

        if "--clean" in args:
            args.remove("--clean")
            clean = True

        failed_files = []
        for arg in args:
            if os.path.isfile(arg) and arg.lower().endswith(".mp3"):
                if clean:
                    clean_mp3(arg)
                else:
                    analyze_files(arg, force)
            elif os.path.isdir(arg):
                files = list_mp3s(arg)
                try:
                    if clean:
                        clean_mp3(files)
                    else:
                        analyze_files(files, directory_name=arg, force=force)
                except KeyboardInterrupt:
                    sys.exit(0)
                except Exception:
                    traceback.print_exc()
            elif arg.lower().endswith(".exe"):
                pass
            else:
                print("Unrecognized argument: %s" % arg)

        if len(failed_files) > 0:
            print("Could not analyze:")
            for file in failed_files:
                print(file)


if __name__ == "__main__":
    main()
