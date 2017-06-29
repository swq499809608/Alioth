#-*-coding:utf-8-*-
import re
import os
import urllib.request

class spider():
    def __init__(self, weburl, webheader=None):
        self.weburl = weburl
        self.webheader = webheader

    def save_file(self, data, file_save_path):  
        f_obj = open(file_save_path, 'wb')
        f_obj.write(data)  
        f_obj.close()

    def get_webpage(self, file_save_path):
        req = urllib.request.Request(url=self.weburl, headers=self.webheader)
        webPage=urllib.request.urlopen(req)
        PageCode = webPage.read()
        self.save_file(PageCode, file_save_path)

    def get_images(self, file_save_path):

        #issues: can't save images
