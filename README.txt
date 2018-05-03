To run the project there has to be installed Python 3 and few modules that can be installed by:

    C:\MMP>pip install -r requirements.txt

To run test:
    C:\MMP\tests>nosetests

to run with coverage:
    C:\MMP\tests>nosetests --with-coverage --cover-erase --cover-package=markets --cover-html

    to run integration tests:
    C:\MMP\tests>behave


How to train models:

    C:\MMP>python manage.py dropdb
    Are you sure you want to lose all your data [n]: y

    C:\MMP>python manage.py initdb

    C:\MMP>python manage.py demo
    1173
    Accuracy 0.5379694250581588 (0.6033911577827612) and zeroR 0.4177323103154305
    Accuracy 0.3867855144891178 (0.6534794778834259) and zeroR 0.36251236399604353
    Model build for USD
    1121
    Accuracy 0.5262748026488129 (0.5885628414586801) and zeroR 0.360392506690455
    Accuracy 0.38071471613874475 (0.66204881484711) and zeroR 0.39317507418397624
    Model build for EUR
    1081
    Accuracy 0.5116646977939098 (0.59081204621171) and zeroR 0.39777983348751156
    Accuracy 0.3901244035880685 (0.6739201798395122) and zeroR 0.3600395647873393
    Model build for MEX

Then to run a webpage:

    C:\MMP>python manage.py run
     * Restarting with stat
     * Debugger is active!
     * Debugger PIN: 826-654-524
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
