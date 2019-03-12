Explain the following things in README of Github
DISCLAIMER:
FINAL MODEL IS "best_model2_ep_50.h5"
DOWNLOAD THE DATA FROM THE FOLLOWING: https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003




    1) Explain in detail the process of feature extraction including the normalization? [ MY APPROACH ]
	I have used three main features:
		1. Glove vector representation of words
		2. Casing of the words. These had the following categories:
			1. Numeric
			2. all_lower
			3. all_upper
			4. initial_upper (Initial charcater is upper case)
			5. Mainly_numeric( If more than half of the string is numeric)
			6. Contains some digits in the string
			7. others
		3. Parts of Speech tag of the respective tag.
	Some Observations and normalizations:
		1. It has been obsrved that each punctuation is given a separate POS tag - the punctuation it self. This has been normalized to "PUNCT" tag.
		2. After normalization there are a total of 40 POS tags, including the "padded_tag" for padded words.
		3. There are a total of 9 classes. The classes are as follows:
			['B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']
		From thhe data, it has been onserved that, the average sentence/ instance length is 14.5 words. This is computed from the function  "average_sent_len(data)". So, the maximum length of the sentence has been set as 15 indivdual words. If the sentence is shorter, the remaining words will be padded with  zero vectors. For the padded tokens, the pos and case feature would be "PADDED_TOKEN". Similarily, there has been addition to the label set, an extra label for padded words - "PADDED_LABEL". So, now there are a total of 10 classes.

Architecture of the model is as follows:
	I have used a bidirectional LSTM  for training the data and  built and  NER system. Implemented with Keras.
	There is a glove embedding layer(word-leve input - 50 dimensions), case embedding layer for the word input( 8 dimensions)  and a POS layer for the words ( 40 dimensions)
	Then there is a concatenate layer, that concatenates all this input and feeds to a Bidirectional LSTM.
	The size of lSTM is 200 units. Output of a bidirectional LSTM, keeping the return_sequences argument "True" is 400 in size.
	After the above normalizations, the model has been trained for 50 epochs, with a batch size of 149. (Obtained by splitting the training data into  equal batches)		
	

   

    2) Describe the hyperparameter choices?
	Following are the hyperparamters that I have experimented with:
		1. LSTM cell size (dimension from output from LSTM)
		2. Tried with making the return sequences of LSTM "Flase" and then  with "True" as well. "True" has given  better performance, but there has been no significant increase.
		3. Experimented with different batch sizes and used adam optimizer ( Ref: A blog said adam converges faster than SGD)
		4. The glove vector representation has been limited to 50 dimensions in all the experiments.
		5. Learning rate is set to 0.001


    3) Apart from LSTM if given a choice what model do you think works better? and why?
	If not an LSTM I would go for either an HMM or a CRF. Reasons being the following:

	1. In the task of named entity recognition, or be it any other langauge related model, modeling sequence is very important. HMMs and CRFs take care of this.
	2. Onr problem with them is they only have context information about the previous states and not about the future states. (But, A BILSTM can do it both ways - Implemented here)
		A small example:
			1. She exclaimed, "Teddy Bears are my favourrite!"
			2. He screamed "Teddy Roosevelt is a great president!"
		To, tag whether Teddy is a "person", its not enough to look at the past information, here it is important to look forward as well.



    4) How does the batch size affect your model?
	I have observed that increasing the  batch size has improved my model performance and  probably this is because when the batch size is big, it estimates the gradient update measures better and then the algorithm takes bigger steps to  get to the minima, but when the batch size gets even bigger, this can be a problem as it may skip the minima.

    

     5) Report Recall, precision, and F1 measure
	Overall Average Recall, Precision and F1 score across classes is:
	Precision: 84.78
	Recall: 87.12
	F1-score: 86

    	        precision    recall  f1-score   support

       B-ORG       0.88      0.87      0.88      1372
           O       0.99      0.99      0.99     26053
      B-MISC       0.70      0.80      0.75       512
       B-PER       0.94      0.94      0.94      1258
       I-PER       0.97      0.98      0.98       833
       B-LOC       0.90      0.93      0.92      1329
       I-ORG       0.88      0.79      0.83       608
      I-MISC       0.60      0.65      0.62       153
       I-LOC       0.77      0.89      0.83       204

After Collapsing the above 9 to 5 classes:
	        
		precision    recall  f1-score   support

         ORG       0.90      0.87      0.88      1980
           O       0.99      0.99      0.99     26053
        MISC       0.69      0.78      0.73       665
         PER       0.96      0.96      0.96      2091
         LOC       0.89      0.93      0.91      1533
Over all average measures:
	Precision: 88.6
	Recall: 90.6
	F-score: 89.4

Future Scope and Discussion

There is a huge class imbalance, that effects the performace of the classifier.
The class statistics are provided in the python notebook.
Character level information can be incorporated ny using char level LSTMS. This can handle out of vocabulary words and this can be extremely useful as the named entities are proper nouns and unseen names can ALWAYS occur.
A stemmer and  a lemmatizer can also be used in the pre-processing steps.
	
