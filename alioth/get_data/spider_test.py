from alioth.spider import spider

# test spider
webheader = {  
	'Connection': 'Keep-Alive',  
	'Accept': 'text/html, application/xhtml+xml, */*',  
	'Accept-Language': 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',  
	'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',  
	'Host': 'www.douban.com',  
	'DNT': '1'  
}  
sp = spider(weburl='http://www.douban.com/',
            webheader=webheader)
sp.get_webpage(file_save_path='./webpage_test.txt')
sp.get_images(file_save_path='D:/images_test/')