{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "A1_newVersion.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/hassthomas/DataScienceAss1/blob/master/A1_newVersion.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vTs11WubmRhV",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# download data (-q is the quiet mode)\n",
        "! wget -q https://www.dropbox.com/s/lhb1awpi769bfdr/test.csv?dl=1 -O test.csv # download test data from dropbox and store in file\n",
        "! wget -q https://www.dropbox.com/s/gudb5eunj700s7j/train.csv?dl=1 -O train.csv # download train data from dropbox and store in file\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jrsga6qkouO1",
        "colab_type": "code",
        "outputId": "b36c7a19-afd3-427f-9215-1a8b7d84e543",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 421
        }
      },
      "source": [
        "import pandas as pd                                     # import library for data processing, CSV file input/output and data manipulation\n",
        "\n",
        "Xy_train = pd.read_csv('train.csv', engine='python')    # read training data from file and store in variable\n",
        "\n",
        "X_train = Xy_train.drop(columns=['price_rating'])       # store training data without the output in variable\n",
        "y_train = Xy_train[['price_rating']]                    # store training data output in variable\n",
        "\n",
        "print('traning', len(X_train))                          # print length (how many columns) of training data\n",
        "Xy_train.price_rating.hist()                            # Make a histogram of the tran data frame’s based on the price rating to get a better understanding how the data looks like\n",
        "\n",
        "X_test = pd.read_csv('test.csv', engine='python')       # read training data from file and store in variable\n",
        "testing_ids = X_test.Id                                 # store the IDs of testing data in variable\n",
        "print('testing', len(X_test))                           # print length (how many columns) of testing data\n",
        "\n",
        "Xy_train.head()                                         # show first 5 columns\n",
        "Xy_train.describe()                                     # give further information about the data\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "error",
          "ename": "EmptyDataError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mStopIteration\u001b[0m                             Traceback (most recent call last)",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/pandas/io/parsers.py\u001b[0m in \u001b[0;36m_infer_columns\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   2592\u001b[0m                 \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2593\u001b[0;31m                     \u001b[0mline\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_buffered_line\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2594\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/pandas/io/parsers.py\u001b[0m in \u001b[0;36m_buffered_line\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   2775\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2776\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_next_line\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2777\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/pandas/io/parsers.py\u001b[0m in \u001b[0;36m_next_line\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   2872\u001b[0m             \u001b[0;32mwhile\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2873\u001b[0;31m                 \u001b[0morig_line\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_next_iter_line\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrow_num\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpos\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2874\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpos\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/pandas/io/parsers.py\u001b[0m in \u001b[0;36m_next_iter_line\u001b[0;34m(self, row_num)\u001b[0m\n\u001b[1;32m   2932\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2933\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mnext\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2934\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mcsv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mStopIteration\u001b[0m: ",
            "\nDuring handling of the above exception, another exception occurred:\n",
            "\u001b[0;31mEmptyDataError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-2-bdae3e0e0b40>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mXy_train\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'train.csv'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mengine\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'python'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0mX_train\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mXy_train\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'price_rating'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0my_train\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mXy_train\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'price_rating'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/pandas/io/parsers.py\u001b[0m in \u001b[0;36mparser_f\u001b[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)\u001b[0m\n\u001b[1;32m    683\u001b[0m         )\n\u001b[1;32m    684\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 685\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0m_read\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    686\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    687\u001b[0m     \u001b[0mparser_f\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__name__\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/pandas/io/parsers.py\u001b[0m in \u001b[0;36m_read\u001b[0;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[1;32m    455\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    456\u001b[0m     \u001b[0;31m# Create the parser.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 457\u001b[0;31m     \u001b[0mparser\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mTextFileReader\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfp_or_buf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    458\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    459\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mchunksize\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0miterator\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/pandas/io/parsers.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[1;32m    893\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"has_index_names\"\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mkwds\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"has_index_names\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    894\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 895\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_make_engine\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mengine\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    896\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    897\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/pandas/io/parsers.py\u001b[0m in \u001b[0;36m_make_engine\u001b[0;34m(self, engine)\u001b[0m\n\u001b[1;32m   1145\u001b[0m                     \u001b[0;34m' \"python-fwf\")'\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mengine\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mengine\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1146\u001b[0m                 )\n\u001b[0;32m-> 1147\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_engine\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mklass\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1148\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1149\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_failover_to_python\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/pandas/io/parsers.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, f, **kwds)\u001b[0m\n\u001b[1;32m   2308\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnum_original_columns\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2309\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munnamed_cols\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2310\u001b[0;31m         ) = self._infer_columns()\n\u001b[0m\u001b[1;32m   2311\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2312\u001b[0m         \u001b[0;31m# Now self.columns has the set of columns that we will process.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/pandas/io/parsers.py\u001b[0m in \u001b[0;36m_infer_columns\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   2613\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2614\u001b[0m                     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnames\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2615\u001b[0;31m                         \u001b[0;32mraise\u001b[0m \u001b[0mEmptyDataError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"No columns to parse from file\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2616\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2617\u001b[0m                     \u001b[0mline\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnames\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mEmptyDataError\u001b[0m: No columns to parse from file"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3vFDJ3lNmE6P",
        "colab_type": "code",
        "outputId": "ea3e89a7-c3e7-45cb-d567-3679a8209a6a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 173
        }
      },
      "source": [
        "# model training and tuning\n",
        "import numpy as np  # library for linear algebra\n",
        "from sklearn.compose import ColumnTransformer # to transform columns seperately\n",
        "from sklearn.datasets import fetch_openml # Fetch dataset from openml by name or dataset id.\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.impute import SimpleImputer # for completing missing values\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder # standarsize data by removing the mean and scaling to unit variance, encode categorial features\n",
        "from sklearn.linear_model import LogisticRegression # to apply the logistic regression\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV  # to split the data into test and train\n",
        "from xgboost.sklearn import XGBClassifier #to apply XGBoost\n",
        "\n",
        "np.random.seed(0) # generate random seed for generating consintent testing\n",
        "\n",
        "# ADD MORE FEATURES TO INCREASE INFORMATION IN MODEL\n",
        "numeric_features = ['bedrooms', 'review_scores_location', 'accommodates', 'beds', \n",
        "                    'host_listings_count', 'minimum_nights', 'maximum_nights', 'number_of_reviews', \n",
        "                    'review_scores_rating','review_scores_accuracy', 'review_scores_cleanliness',\n",
        "                    'review_scores_checkin', 'review_scores_communication','review_scores_location',\n",
        "                    'review_scores_value', 'reviews_per_month']] # select the numeric features\n",
        "numeric_transformer = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='median')),\n",
        "    ('scaler', StandardScaler())])  #Sequentially apply a list of transforms and a final estimator; SimpleImputer: replaces missing value with median for numeric_features\n",
        "                                    #StandardScaler: Normalize the data \n",
        "\n",
        "categorical_features = [ 'property_type', 'is_business_travel_ready', 'room_type','host_response_time', \n",
        "                        'host_is_superhost', 'host_identity_verified', 'neighbourhood_cleansed',\n",
        "                        'instant_bookable', 'cancellation_policy', 'require_guest_phone_verification'] ] # select the categorical features \n",
        "categorical_transformer = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\n",
        "    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  #OneHotEncoder: transform into vector\n",
        "\n",
        " #   Applies transformers to columns\n",
        "preprocessor = ColumnTransformer(\n",
        "    transformers=[\n",
        "        ('num', numeric_transformer, numeric_features),\n",
        "        ('cat', categorical_transformer, categorical_features)]) # transforms data\n",
        "\n",
        "\n",
        "#set training and testing data with the chosen numerical and categorical features; so that the others are not noted anymore\n",
        "X_train = X_train[[*numeric_features, *categorical_features]] \n",
        "X_test = X_test[[*numeric_features, *categorical_features]]\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Fitting 5 folds for each of 4 candidates, totalling 20 fits\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "[Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.\n",
            "[Parallel(n_jobs=2)]: Done  20 out of  20 | elapsed:   41.0s finished\n",
            "/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_label.py:235: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
            "  y = column_or_1d(y, warn=True)\n",
            "/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_label.py:268: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
            "  y = column_or_1d(y, warn=True)\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "best score 0.6945360101828082\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1U3ZEswV5rj3",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#--------------- XGB CLASSIFIER ---------------\n",
        "#create Pipeline that applies preprocessor which was defined above and then uses xgb as classifier\n",
        " \n",
        "pipeline1 = Pipeline(steps=[('preprocessor', preprocessor),\n",
        "                      ('regressor', XGBClassifier(\n",
        "                          objective='multi:softmax', seed=1))])\n",
        "\n",
        "# `__` denotes attribute \n",
        "# (e.g. regressor__n_estimators means the `n_estimators` param for `regressor`\n",
        "#  which is our xgb)\n",
        "param_grid = {\n",
        "    'preprocessor__num__imputer__strategy': ['mean'],\n",
        "    'regressor__n_estimators': [50, 100], # test first 50 and then 100 estimators\n",
        "    'regressor__max_depth':[10, 20]     # use a maximum test depth of 10 and 20\n",
        "}\n",
        "\n",
        "\n",
        "# GridSearchCV: implements a “fit” and a “score” method. Find the best parameters to use.\n",
        "#   Do Cross-validation 5 times\n",
        "#   n_jobs = 2 means running two jobs in parallel\n",
        "#   verbose - give messages\n",
        "#   evaluate the test set on accuracy\n",
        "grid_search = GridSearchCV(\n",
        "    pipeline1, param_grid, cv=5, verbose=3, n_jobs=2, \n",
        "    scoring='accuracy')\n",
        "\n",
        "grid_search.fit(X_train, y_train) #fitting model with data\n",
        "print('best score {}'.format(grid_search.best_score_))  #printing the accuracy\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "OsqND8zm6ayD",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#--------------- RANDOM FOREST ---------------\n",
        "\n",
        "from sklearn.ensemble import RandomForestClassifier #import needed library\n",
        "\n",
        "#create Pipeline that applies preprocessor which was defined above and then uses RandomForest as classifier\n",
        "pipeline2 = Pipeline(steps=[('preprocessor', preprocessor), ('rf', RandomForestClassifier())])\n",
        "\n",
        "#control flexibility by settings\n",
        "paramForest = {\"rf__n_estimators\": [500],\n",
        "             \"rf__max_depth\": [60],\n",
        "             \"rf__min_samples_split\": [8],\n",
        "             \"rf__min_samples_leaf\": [1]\n",
        "             }\n",
        "\n",
        "rf_gs = GridSearchCV(pipeline2, paramForest, cv=5, verbose=3, n_jobs=2, scoring='accuracy')  \n",
        "\n",
        "\n",
        "rf_gs.fit(X_train, y_train)  #fitting model with data\n",
        "print('best score {}'.format(rf_gs.best_score_)) #print accuracy\n",
        "\n",
        "\n",
        "rf_best = rf_gs.best_estimator_ #save the best model\n",
        "print(rf_gs.best_params_) #print best parameters"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BATFrYDm8SIa",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#--------------- MLP | Neural Network ---------------\n",
        "\n",
        "from sklearn.neural_network import MLPClassifier #import needed library\n",
        "\n",
        "#create Pipeline that applies preprocessor which was defined above and then uses MLP as classifier\n",
        "pipeline3 = Pipeline(steps=[('preprocessor', preprocessor), ('nn', MLPClassifier())])\n",
        "\n",
        "#control flexibility by settings\n",
        "paramNeural = {\"nn__activation\": [\"logistic\"],\n",
        "             \"nn__solver\": [\"adam\"],\n",
        "             \"nn__alpha\": [0.04],\n",
        "             \"nn__max_iter\": [300],\n",
        "             \"nn__learning_rate\": ['adaptive']\n",
        "             }\n",
        "\n",
        "nn_gs = GridSearchCV(pipeline3, paramNeural, cv=5, verbose=3, n_jobs=2, scoring='accuracy')\n",
        "\n",
        "nn_gs.fit(X_train, y_train) #fitting model with data\n",
        "print('best score {}'.format(nn_gs.best_score_)) #print accuracy\n",
        "\n",
        "nn_best = nn_gs.best_estimator_ #save the best model\n",
        "print(nn_gs.best_params_) #print best parameters\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "n7Nh_mv-9pQX",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#--------------- USE ENSAMBLED METHODS TO COMBINE (VOTING CLASSIFIER) ---------------\n",
        "\n",
        "from sklearn.ensemble import VotingClassifier #import needed library\n",
        "\n",
        "estimators = [('xgb',xgb_best), ('nn', nn_best), ('rf', rf_best)] #create dictionary by selecting the 3 used models\n",
        "\n",
        "\n",
        "ensemble = VotingClassifier(estimators, voting='hard') #create voting classifier to combine them and set voting to hard\n",
        "\n",
        "ensemble.fit(X_train, y_train) #fitting model with data\n",
        "print('best score final: ', ensemble.score(X_train, y_train)) #print final accuracy"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PF6WrzdKmJ97",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Prediction & generating the submission file\n",
        "y_pred = ensemble.predict(X_test)\n",
        "pd.DataFrame(\n",
        "    {'Id': testing_ids, 'price_rating':y_pred}).to_csv('final_submission.csv', index=False)"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}