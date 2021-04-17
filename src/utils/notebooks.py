from typing import List
from IPython.core.display import HTML
import pandas as pd


def display_side_by_side(frames: List[pd.DataFrame], titles: List[str] = None, padding: int = 5) -> None:

    content = ""
    for i, frame in enumerate(frames):
        title = "" if not titles else titles[i]
        content = f"""{content}
                      <div style='display:block;float:left;padding:0 {padding}px 0 0;text-align:center'>
                         <h3>{title}</h3><br />
                         {frame.to_html()}
                      </div>
                    """
    display(HTML(content))
