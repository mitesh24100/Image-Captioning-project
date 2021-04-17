from flask import Flask, render_template, redirect, request
import caption

app = Flask(__name__)

@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/', methods = ['POST'])
def marks():
	if request.method == 'POST':
		f = request.files['userfile']
		path = "./static/{}".format(f.filename)
		f.save(path)
		caption2 = caption.caption_this_image(path)
		

	return render_template("index.html", photo = path, caption1 = caption2)

if __name__ == '__main__':
	app.run(debug = True)
