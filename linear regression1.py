{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "855d2b17",
   "metadata": {},
   "source": [
    "# SIMPLE LINEAR REGRESSION WITH PYTHON"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e0d8d6c7",
   "metadata": {},
   "source": [
    "## Predicting percentage of astudent based on number of hours studied"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a946e99f",
   "metadata": {},
   "source": [
    "### Author : Bandi Swathi Yadav"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8344d10a",
   "metadata": {},
   "source": [
    "**Import libraries required**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "be3fe34c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "from sklearn import linear_model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "46a9d46f",
   "metadata": {},
   "source": [
    "**Getting the data**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e6534a9f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1.5</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>9.2</td>\n",
       "      <td>88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>5.5</td>\n",
       "      <td>60</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8.3</td>\n",
       "      <td>81</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>2.7</td>\n",
       "      <td>25</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30\n",
       "5    1.5      20\n",
       "6    9.2      88\n",
       "7    5.5      60\n",
       "8    8.3      81\n",
       "9    2.7      25"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url=(\"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv\")\n",
    "df=pd.read_csv(url)\n",
    "df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7cfc18ef",
   "metadata": {},
   "source": [
    "**Plotting the graph**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "305c4d70",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.xlabel('Hours')\n",
    "plt.ylabel('Scores')\n",
    "plt.scatter(df.Hours,df.Scores,color='red',marker='+')\n",
    "line=reg.coef_*df.Hours+reg.intercept_\n",
    "plt.plot(df.Hours,line);"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d279eaaa",
   "metadata": {},
   "source": [
    "**Dropping Y column**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "38bd3641",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     21\n",
       "1     47\n",
       "2     27\n",
       "3     75\n",
       "4     30\n",
       "5     20\n",
       "6     88\n",
       "7     60\n",
       "8     81\n",
       "9     25\n",
       "10    85\n",
       "11    62\n",
       "12    41\n",
       "13    42\n",
       "14    17\n",
       "15    95\n",
       "16    30\n",
       "17    24\n",
       "18    67\n",
       "19    69\n",
       "20    30\n",
       "21    54\n",
       "22    35\n",
       "23    76\n",
       "24    86\n",
       "Name: Scores, dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_df=df.drop('Scores',axis='columns')\n",
    "new_df\n",
    "Scores=df.Scores\n",
    "Scores"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4559eadd",
   "metadata": {},
   "source": [
    "**Fitting the data into the model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b43280c2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg=linear_model.LinearRegression()\n",
    "reg.fit(new_df,Scores)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "430fa91b",
   "metadata": {},
   "source": [
    "** predicting the Score by giving Hours studied**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "886f418b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([92.90985477])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.predict([[9.25]])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7f681bd",
   "metadata": {},
   "source": [
    "**Checking the predited Score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "a5a68ed0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([9.77580339])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.coef_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b1b7fc95",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.483673405373196"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.intercept_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "7e9d83b0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "92.9098547628732"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "9.77580339*9.25+2.483673405373196"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6e1a864d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "celltoolbar": "Raw Cell Format",
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
