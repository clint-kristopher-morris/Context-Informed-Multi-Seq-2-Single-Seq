### BackGround
___

<video src="https://i.imgur.com/hcRuFBH.mp4" width="640" height="360" controls></video>


### Installation Guide
___
The installation steps outlined below assume that you have already installed the Anaconda package manager on your device.
#### Step 1: Set Directory
To navigate to the directory where the app is located, you can use the "cd" command in your terminal or command prompt. Here's an example command:
```commandline
cd /path/to/app/directory/RP2010_application
```
#### Step 2: Create Environment with Conda
```commandline
conda create --name RP2010app --file requirements.txt
conda activate RP2010app
```
#### Step 3: Run Application
```commandline
streamlit run app.py
```
