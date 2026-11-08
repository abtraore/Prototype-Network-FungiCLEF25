"""open_clip_train entry point that tolerates the truncated JPEGs in FungiTastic."""

import sys

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from open_clip_train.main import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
