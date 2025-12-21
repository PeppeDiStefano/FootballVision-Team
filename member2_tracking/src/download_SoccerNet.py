from SoccerNet.Downloader import SoccerNetDownloader

myDownloader = SoccerNetDownloader(LocalDirectory="./SoccerData")



myDownloader.downloadDataTask(task="tracking-2023", split=["train"])