#importing libraries

import tensorflow as tf
import numpy as np
import web
import os
import time


render = web.template.render('./templates/')
urls = ('/', 'index')
app = web.application(urls, globals())


#define index class
class index:
	def GET(self): 
		#display index.html template				
		return render.index(None, None, None)

	def POST(self):
		return render.index(None, None, None)

if __name__=='__main__':

	app.run()
	
 		
