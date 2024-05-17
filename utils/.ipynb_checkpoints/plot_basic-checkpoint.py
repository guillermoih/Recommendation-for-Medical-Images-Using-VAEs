# +
import json

import numpy as np
# -

with open('/workspace/Guille/MOC-AE/MOC-AE_Code/config.json', 'r') as f:
    config = json.load(f)


def show_case(X, y, names, idx, ax):
    ax.imshow(X[idx].reshape(config["padchest"]["image"]["img_height"],
                             config["padchest"]["image"]["img_width"]),
              cmap='gray')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(names[np.argmax(y[idx])])
