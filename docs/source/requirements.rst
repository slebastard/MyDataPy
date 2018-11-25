Project requirements
====================

Basic requirements
----------------
scipy==0.19.1
matplotlib==2.0.2
numpy==1.13.1
pandas==0.20.3
protobuf==3.6.1


Extras (optimization, vizualization, ...)
-----------------------------------------
cvxopt==1.1.9
	provides access to CVXOpt performant optimization methods to solve optim problems (kernel SVM uses it) 
google_api_python_client==1.7.4 & gspread==0.6.2
	if you wish to import dataset directly from GoogleSheets (useful for some colaborative projects)
