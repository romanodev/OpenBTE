import webbrowser 
import pathlib


def main():
 """Open documentation"""

 pp = str(pathlib.Path(__file__).parent.resolve()) + '/../docs/build/html/index.html'

 webbrowser.open(pp)

