# Inteligent-Paper
Generate Pix2Pix drawing using data from touchboard sensor
<h2> Idea </h2>
Touchboards are interactive devices that use electric paint to detect when someone touches them and then sends a signal to a connected device. When you connect paint to the sensors, the paint becomes the sensor. You can then touch wood or glass to trigger the sensor. There are lots of possibilities for creating interactive projects. Bare conductive is built with Arduino and has 13 input signals, meaning you can do 13 different things interactively. This is great for simple interaction. If you want to create a keyboard or musical instrument, though, 13 signals might not be enough. My idea was to connect two devices, one as the X axis and second as the Y axis. This gives us 13x13, which makes 169 points. If two sensors recognize a touch within a short amount of time, I consider it a true touch and set the signal. I thought that 13x13 might be even enough for a simple scratch, though the quality would be quite poor. However, some generative algorithms, such as Pix2Pix, could help to improve this.<br>
<br>
Source:<br>
<ul>
  <li> <a href="https://www.bareconductive.com/"> Bare Conductive</a></li>
  <li>  <a href="https://www.bareconductive.com/pages/what-is-bare-conductive">What Is Bare Conductive</a></li>
  <li>  <a href="https://www.bareconductive.com/pages/touch-board-get-started">Touch Board: Get Started – Bare Conductive</a></li>
</ul>  
<h2> Pix2Pix </h2>
<br>
Pix2pix is an image-to-image translation algorithm. It allows you to turn a source image into a target image using a deep learning model. The model uses a combination of convolutional neural networks to learn the mapping between the two images. It can be used to create realistic images from sketches, turn aerial photos into maps, or turn day photos into night photos. Now there is a question about how to access a dataset for simple drawing. Fortunately, there is Quick Draw, which provides an easy way to do so.<br>

![10103_AB](https://user-images.githubusercontent.com/40691316/215875303-5ba19288-9ca5-444a-b393-819a0291b6aa.jpg)
<br>
Picture from Kaggle dataset available at <a href="https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset">here.</a> <br>
<br>
Source:
<ul>
  <li> <a href="https://www.tensorflow.org/tutorials/generative/pix2pix">Pix2pix: Image-to-image translation | TensorFlow Core</a></li>
  <li>  <a href="https://towardsdatascience.com/pix2pix-869c17900998">Pix2Pix by Connor Shorten </a></li>
  <li>  <a href="https://ml4a.github.io/guides/Pix2Pix/">Pix2Pix (ml4a.github.io)</a></li>
</ul>  
<h2> Quick Draw </h2>
<br>
Google's project Quick Draw is an online game in which players are presented with a simple doodle and they must draw it in the allotted time. The game records the drawings and stores them in a dataset. The dataset contains over 50 million drawings across 345 categories, such as animals, objects, and people. This data can be used to train machine learning models to recognize doodles and to create new applications.<br>
<br>
Source:<br>
<ul>
  <li> <a href="https://quickdraw.withgoogle.com/">Quick, Draw! (quickdraw.withgoogle.com)</a></li>
  <li>  <a href="https://experiments.withgoogle.com/quick-draw">Quick, Draw! by Google Creative Lab</a></li>
  <li>  <a href="https://thecodingtrain.com/challenges/122-quick-draw">Quick, Draw! / The Coding Train</a></li>
</ul>  
<h2> Data preparation </h2>
<br>
For training purposes, I need to prepare pairs of pictures: the left one is the target drawing, and the right one is the source, which is the expected signal from the touchboard. I have used the Quick Draw dataset in NDJSON format and converted it to a picture using the Python package Cairo.<br>
<br>

![6456634324811776](https://user-images.githubusercontent.com/40691316/215863505-edfd764a-0e36-47d9-9091-1deb0f0471b8.jpg)

As the second step, I created a mask to match the pixels from the drawing with the X and Y axes. To do this, I simply used the logical_and function from NumPy on two arrays.<br>
<br>
![6456634324811776 - Copy](https://user-images.githubusercontent.com/40691316/215863552-c89f43ba-f5dd-45a8-b6d7-2d42690065f1.jpg)
<h2> Training </h2>
I have used the code provided on the TensorFlow tutorial page. You can see it <a href="https://www.tensorflow.org/tutorials/generative/pix2pix">here.</a> I required approximately 600 pairs of images in order to obtain a satisfactory result. Due to the inconsistency of the quality of quickDraw images, I had to manually verify a lot of them before I was able to create a satisfactory training dataset.<br>
<br>
<h2> Data from sensor </h2>
<br>
The Touchboard communicates with a computer using a serial port. There is a standard program available on the BareConductive website that can be uploaded to the device. When an electrode is touched, the number of sensors is sent over the serial port. On the other side, there is a Python script listening to both devices (X-axis and Y-axis). When both devices are touched within 0.1 seconds of each other, it is considered true.<br>  
<h2> Putting all together </h2>
Test test
<h2> Remaining work </h2>
<br>
<ul>
  <li>There is still a remaining "hardware" part that needs to be done with precision in order to avoid any interference between the two touch boards and to prevent any shaking from occurring.</li>
  <li>To achieve stunning results, the quality of training and testing data should be improved. Ideally, more advanced datasets of drawing should be available.</li>
 
</ul> 
