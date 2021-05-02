import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.core.display import HTML, display


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


def plot_word_cloud(sentences: List[str], figure_size: Tuple[int, int] = None) -> None:
    wordcloud = WordCloud(background_color="white",  mode='RGBA', scale=10).generate(" ".join(sentences))
    plt.figure(figsize=figure_size)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
