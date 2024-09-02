{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f3af8922-1dbb-436e-bbfc-925f5f806f79",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-02 10:55:03.794 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\ProgramData\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import streamlit as st\n",
    "st.title('Welcome to Streamlit!')\n",
    "st.write('Our first DataFrame')\n",
    "st.write(\n",
    "    pd.DataFrame(\n",
    "        {\n",
    "            'x':[1,0,2,3,4,6],\n",
    "            'y':[2,4,5,6,8,9]\n",
    "        }\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca40bf3c-cf2a-405e-a782-4ebdcca3472d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
