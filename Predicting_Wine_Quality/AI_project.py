#CODE WRITTEN BY: ABISHEK LAKANDRI



import numpy as np
import pandas as pd
from io import StringIO 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Split the data for cross-validation(test size = 25% and training size = 75 % of total data)
def split_testTrain(X, y, y_bin):
	# Splitting the original data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
	
	#Splitting the binary classification output data
	X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(X, y_bin, 
		test_size=.25, random_state=0)
	return [X_train, y_train, X_bin_train, y_bin_train, X_bin_test, y_bin_test]



def print_func(model_name, accuracy):
	print(f"\n\n\n**********{model_name.upper()} Model**********")
	print(f'\nThe accuracy of {model_name} is: {accuracy * 100}%')
	print(f'The classification report for {model_name} is:')


def print_selected_model(model_name):
	print(f'\nNOTE: Among all the three models, {model_name.upper()} has the highest accuracy for classification.\n'
		f'So we will choose {model_name.upper()} for classifying the new wine samples.')



# Prepare Decision Tree model
def decisiontree_model(data):
	model1 = DecisionTreeClassifier(random_state=1)
	model1.fit(data[2], data[3])
	y_predict = model1.predict(data[4])
	accuracy_model1 = round(accuracy_score(data[5], y_predict), 2)
	print_func('Decision Tree', accuracy_model1)
	print(classification_report(data[5], y_predict))
	return model1, accuracy_model1




# Prepare Random Forest model
def randomforest_model(data):
	model2 = RandomForestClassifier(random_state=1)
	model2.fit(data[2], data[3])
	y_predict = model2.predict(data[4])
	accuracy_model2 = round(accuracy_score(data[5], y_predict), 2)
	print_func('Random Forest', accuracy_model2)
	print(classification_report(data[5], y_predict))
	return model2, accuracy_model2




# Prepare Gradient Boosting model
def gradboost_model(data):
	model3 = GradientBoostingClassifier(random_state=1)
	model3.fit(data[2], data[3])
	y_predict = model3.predict(data[4])
	accuracy_model3 = round(accuracy_score(data[5], y_predict), 2)
	print_func('Gradient Boosting', accuracy_model3)
	print(classification_report(data[5], y_predict))
	return model3, accuracy_model3




def main():
	df = pd.read_csv("winequality-red.csv")
	print(len(df))
	print(type(df))

	# Create binary version of target variable
	df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]

	# Separate feature variables and target variable
	X = df.drop(['quality','goodquality'], axis = 1)
	y = df['quality']
	y_bin = df['goodquality']

	# Normalize feature variables
	X_features = X
	X = StandardScaler().fit_transform(X)

	#Split the data for cross validation
	test_train_data = split_testTrain(X, y, y_bin)


	# Model 1: Decision Tree
	model1, accuracy_model1 = decisiontree_model(test_train_data)
	model1.fit(test_train_data[0], test_train_data[1])


	# Model 2: Random Forest
	model2, accuracy_model2 = randomforest_model(test_train_data)
	model2.fit(test_train_data[0], test_train_data[1])


	# Model 3: Gradient Boosting
	model3, accuracy_model3 = gradboost_model(test_train_data)
	model3.fit(test_train_data[0], test_train_data[1])

	if (accuracy_model1 > accuracy_model2) and (accuracy_model1 > accuracy_model3):
		model = model1
		print_selected_model('Decision Tree')
	elif (accuracy_model2 > accuracy_model1) and (accuracy_model2 > accuracy_model3):
		model = model2
		print_selected_model('Random Forest')
	else:
		model = model3
		print_selected_model('Gradient Boosting')

	feat_importances = pd.Series(model.feature_importances_, index=X_features.columns)
	sorted_feat = feat_importances.sort_values(ascending=False)
	print(f'\nThe chemical contents of wine inorder of feature importance:')
	print(sorted_feat)
	keys = sorted_feat.keys()
	print('\nThe top four chemical contents(features) are:')
	for i in range(4):
		print(f'{keys[i]}')
		
	wine_features = ['fixed acidity', 'volatile acidity', 'citric acid', 
	'residual sugar', 'chlorides', 'SO2(free)', 'SO2(Total)',
	 'density', 'pH', 'sulphates', 'alcohol']

	selection = input("\nSelect your choice:\n-Press 1 to determine quality of a wine sample.\n-Press any other key to exit.\n")
	while(True):
		if (selection.isdigit() and int(selection) == 1):
			
			while(True):
				try:
					type_data = int(input('1. Upload a file or\n2. Enter the chemical contents manually or\n3. Any other num to quit:\n'))
					break
				except ValueError:
					print('**Numeric entries only.**')

			if type_data == 1:
				file_name = input('Enter full name of the file including .csv (must be in the same folder):\n')
				try: 
					df2 = pd.read_csv(file_name)
					original_qual = df2['quality']
					
				except FileNotFoundError:
					print('File not found. Try again later.')
					break

			elif type_data == 2:
				values = []
				for i in range(len(wine_features)):
					while True:
						try:
							input_data = float(input(f'Enter {wine_features[i]}: \n'))
							break
						except ValueError:
							print('**Must be a numeric value.**')
					values.append(input_data)
				df2 = pd.DataFrame([values], columns = wine_features)
			else:
				break

			length = len(df2)
			new_df = pd.concat([df, df2], ignore_index = True)
			new_df = new_df.drop(['quality','goodquality'], axis = 1)
			new_df = StandardScaler().fit_transform(new_df)

			normalized_values = new_df[-length:]
			df2['quality'] = model.predict(normalized_values)
			df2['overall '] = ['Good' if i >= 7 else 'Bad' for i in df2['quality']]
			if type_data == 1:
				print(f"\nThe samples in {file_name} are classified with the accuracy of {round(accuracy_score(original_qual, df2['quality'])  * 100, 3)}%.")
			print('\n')
			print(df2.to_string())

			selection = input("\nSelect your choice:\n-Press 1 to determine quality of a wine sample.\n-Press any other key to exit.\n")
		else:
			break


if __name__ == "__main__":
	main()