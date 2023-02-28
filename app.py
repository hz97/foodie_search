#!/usr/bin/python
import os
from tkinter import *
import webbrowser

from crawl_data import crawl_data
from data_types import SearchType, RankType
from index import Index
from load_data import load_data
from process_data import process_data

RAW_DATA_PATH = './raw_data'
PROCESSED_DATA_PATH = './processed_data'


class MyApp:
    def __init__(self, parent):
        self.myParent = parent
        self.myContainer1 = LabelFrame(parent, text="Foodie Search")
        self.myContainer2 = Frame(parent)
        self.myContainer1.grid()
        self.myContainer2.grid()

        self.label1 = Message(self.myContainer1, text="")

        self.label1.pack()

        self.entryVariable = StringVar()
        self.textbox = Entry(self.myContainer1)
        self.textbox.pack(side=TOP)
        self.textbox.bind("<Return>", self.submit_search)
        self.textbox.focus_set()

        self.label2 = Label(self.myContainer1)
        self.label2["text"] = ""
        self.label2.pack(side=BOTTOM)

        self.label3 = Message(self.myContainer1, text="")
        self.label3.pack()

        self.button1 = Button(self.myContainer2)
        self.button1["text"] = "J'ai faim!"
        self.button1.pack(side=BOTTOM)
        self.button1.bind("<Button-1>", self.submit_search)

    def submit_search(self, _):
        query = self.textbox.get()
        results = index.search(query, search_type=SearchType.AND, rank_type=RankType.TF_IDF)
        if results:
            webbrowser.open_new(f'https://instagram.com/p/{results[0]}')


if __name__ == '__main__':
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        crawl_data()
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        process_data()

    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        process_data()

    posts = load_data()
    index = Index(posts)
    print(f'Index contains {len(index.id_to_post)} posts')

    root = Tk()
    myapp = MyApp(root)
    root.title('Foodie Search')
    root.mainloop()
