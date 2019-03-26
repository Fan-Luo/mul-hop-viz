<h1>Visualization for Multi-hop Reasoning</h1>

To run it:   

1. git pull

2. **download** the [glove file](https://drive.google.com/file/d/1ZIEjV6IoaDJ_cru-c0uAGp4tXDkBgeWZ/view?usp=sharing) and **unzip** it, and then put in the same folder where *multi-hop-reasoning.py* locates.

3. run *python multi-hop-reasoning.py*

	( I use pytorch_*1.0.0*. If the version is too low, it can be problematic.)

4. go to http://127.0.0.1:5000/annotation

 (You will see a skeleton page first. About half a second, the first example is loaded after Get from the model.)
 
5. Click 'next'    

	It sents the annotaion back to serve so that it gets used by the model, and the model provides next example, whcih would replace the previous one in the browser. 