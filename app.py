from utils import *
from models import RealTimeSentimentAnalyzerW2V, RealTimeSentimentAnalyzerHBRT 

def main():
    inpt = input('What model do you want to use wav2vec2 or hubert \n')

    app = App(inpt)

    if inpt == 'hubert':
        analyzer = RealTimeSentimentAnalyzerHBRT(output_queue=app.queue)
    else:
        analyzer = RealTimeSentimentAnalyzerW2V(output_queue=app.queue)

    def start():
        analyzer.start()
        app.set_status("Running...")

    def stop():
        analyzer.stop()
        app.set_status("Stopped!")

    app.set_start_command(start)
    app.set_stop_command(stop)
    app.protocol("WM_DELETE_WINDOW", lambda: (stop(), app.destroy()))

    app.mainloop()

if __name__ == "__main__":
    main()