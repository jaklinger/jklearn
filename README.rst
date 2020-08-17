jklearn
=======

+-------------+---------------+
| Branch      | Build         |
+=============+===============+
| Master      | |build|       |
+-------------+---------------+
| Development | |build (dev)| |
+-------------+---------------+

Currently a repository which wraps just a single algorithm "Omnislash", which performs
super rough, super fast, divisive hierarchical clustering to large high dimensional data.


Installation
------------

Currently only available from source::

  pip install git+https://github.com/jaklinger/jklearn


Omnislash
---------

You can find a very soft description of the algorithm here_.

.. _here: https://www.nesta.org.uk/blog/omnislash/

Usage:

.. code-block:: python3

     from jklearn.cluster import Omnislash
     import requests
     import pandas as pd
  
     data = []
     r = requests.get("http://cs.joensuu.fi/sipu/datasets/s2.txt")
     for line in r.text.split("\n"):
         if line.strip() == "":
             continue
	 x, y = line.strip().split()
	 data.append({"x":float(x), "y":float(y)})
     df = pd.DataFrame(data)

     omni = Omnislash(50, evr_max=0.75, sample_space_size=1000, n_components_max=2)
     labels = omni.fit_predict(data)
     fig, ax = plt.subplots(figsize=(6, 6))
     ax.scatter(df.x, df.y, c=labels, cmap="tab20")

  
.. |build| image:: https://travis-ci.org/jaklinger/jklearn.svg?branch=master
    :target: https://travis-ci.org/jaklinger/jklearn
    :alt: Build Status (master)

.. |build (dev)| image:: https://travis-ci.org/jaklinger/jklearn.svg?branch=dev
    :target: https://travis-ci.org/jaklinger/jklearn
    :alt: Build Status (dev)	  


