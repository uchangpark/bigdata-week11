#!/usr/bin/env python
# coding: utf-8
# # CHAPTER 8 영문 텍스트 데이터 분석
# ## 8.2 텍스트 데이터 분석
# =============================================================================
# ### 8.2.2 영어 뉴스 데이터 수집
# =============================================================================
# #### O 필요한 패키지 읽어들이기
# - http.client는 데이터를 요청하는 client의 http 프로토콜을 관리하는 패키지
# - Urllib.request는 url을 열고 데이터를 읽어 들이는 패키지<br />
# - Urllib.parse 는 url을 통해 읽어온 데이터를 문법적으로 분석<br />
# - Urllib.error 는 request에서 발생하는 오류 처리<br />
# - Base64는 읽어온 이진 형태의 데이터를 ASCII 형태로 변환<br />
# - Json은 json 스트링이나 파일을 파싱하는 패키지<br />
# - Pandas는 데이터프레임을 사용하기 위한 패키지<br />
# - Numpy는 수학적 연산을 하기 위한 패키지<br />
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json
import pandas as pd
import numpy as np

# =============================================================================
# ### 8.2.3 텍스트 데이터 전처리
# =============================================================================
import nltk # 영문 자연어 처리 패키지 NLTK(Natural Language ToolKit)를 불러들임
df = pd.read_csv('bing_news_shuffle.csv', encoding = 'cp949')
df
# =============================================================================
# # ##### 3.1 토큰 추출(Tokenization)
# =============================================================================
print(df.iloc[0]['description']) # 첫번째 행의 뉴스 내용 출력해봄
"""As the general election ploughs on, a different type of campaign is 
raging in the sleepy constituency of Buckingham. In the territory, 
nestled between Oxford, Aylesbury and Milton Keynes, 
more than 75,000 voters have not been heard in Parliament since 2009."""
# 첫번째 행의 뉴스 내용에서 토큰 추출
tokens = nltk.word_tokenize(df.ix[0]['description']) 
tokens

# - for 문을 사용해 위에서 추출한 tokens의 각 토큰마다 작업 진행
# - len 함수를 사용해 토큰의 길이를 구하고, 
#   1보다 큰 token을 lower함수를 사용해 모두 소문자로 변경
# - 길이가 1보다 크고, 모두 소문자로 변경된 토큰들을 tokens에 저장
tokens = [token.lower() for token in tokens if len(token) > 1]
tokens

# #### Bi-gram
# - NLTK의 bigrams 함수를 앞에서 추출한 tokens에 적용해 bigram을 추출하고 tokens_bigram에 저장
# - for 문을 사용해 tokens_bigram에 저장된 토큰을 하나씩 출력
tokens_bigram = nltk.bigrams(tokens)
for token in tokens_bigram :
    print(token)
# #### Tri-gram
# - NLTK의 trigrams 함수를 앞에서 추출한 tokens에 적용해 trigram을 추출하고 tokens_trigram에 저장
# - for 문을 사용해 tokens_trigram에 저장된 토큰을 하나씩 출력
tokens_trigram = nltk.trigrams(tokens)
for token in tokens_trigram :
    print(token)
    
# =============================================================================
# # ##### 3.2 정제(Cleansing)
# =============================================================================
# - nltk.corpus에 설치된 stopwords를 import
from nltk.corpus import stopwords
# - stopwords중 영어 불용어에 해당하는 english 를 가져와 stopwords에 저장
stop_words = stopwords.words('english')
stop_words

# - for문을 사용해 tokens에 저장된 token들에 작업 진행
# - if not ~ in 구문을 사용해 token이 stopwords에 존재하는지 확인
# - token이 stopwords에 존재하지 않을 경우 tokens_clean에 저장
tokens_clean = [token for token in tokens if not token in stop_words]
tokens_clean

# =============================================================================
# # ##### 3.3 POS tagging
# =============================================================================
# - NLTK의 내장 형태소 분석기 pos_tag를 사용해 tokens에 저장된 토큰들에 
#   대한 품사를 추출 - 각 토큰과 해당 품사를 변수 tokens_tagged에 저장
tokens_tagged = nltk.pos_tag(tokens_clean)
print(tokens_tagged)

# 형태소 tag 확인
# nltk.help.upenn_tagset()
# - 형태소 분석이 된 tokens_tagged의 토큰(word)과 형태소(pos)에 대해 
#   for문을 사용해 작업 진행 - pos가 명사[‘NN’, ‘NNP’]에 해당하는지 확인
# - NN: 명사, NNP: 고유명사,
# - 명사일 경우 word를 tokens_noun에 저장 
tokens_noun = [word for word, pos in tokens_tagged if pos in ['NN', 'NNP']]
print(tokens_noun)

# =============================================================================
# ### 8.2.4. WordCloud
# =============================================================================
# ##### Word cloud package 설치
# - Anaconda Prompt 창에서 python --version
# - http://www.lfd.uci.edu/~gohlke/pythonlibs/#wordcloud 로 가서 
#    알맞은 설치 파일 다운로드
# 1. cp36: python 3.5
# 2. win32: 32bit
# 3. win amd64: 64bit
# 
# - 다운 받은 화일을 Anaconda 설치 폴더 (C:\Users\사용자이름\Anaconda3) 아래 저장
# - Anaconda Prompt에서 위 위치로 이동
# - pip install wordcloud~.whl 입력하면 설치가 됨
# - 시각화 결과를 jupyter 작업창 내에서 보기 위한 설정
# - 데이터 시각화를 지원하는 패키지 pyplot을 import
# - wordcloud를 그리는 모듈 WordCloud, 불용어 리스트 STOPWORDS 
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
# - STOPWORDS를 set 함수를 사용해 변수 stopwords에 저장
# - WordCloud함수를 사용해 word cloud를 그리는 모듈 wc 생성
# - 그림 배경 색상 background_color를 white로 지정
# - 가장 큰 단어의 크기 max_font_size를 100으로 지정
# - 그림 안에 표시될 빈도수가 높은 단어 수 max_words를 50개로 지정
# - 단어 내부 불용어 제거를 위해 stopwords를 stopwords로 지정

stopwords = set(STOPWORDS)
wc = WordCloud(background_color="white", max_font_size=100, max_words=50, stopwords=stopwords)

# - 뉴스 내용인 description 부분을 str 함수를 사용해 string 형태로 변환하고
# - cat 함수를 사용해 문자열들을 한 문자열로 연결
# - sep=‘, ‘를 사용해 각 문자열을 연결할 때 구분자로 콤마를 지정
# - 한 문자열로 합쳐진 뉴스 내용을 text_data에 저장
# - generate 함수를 사용해 문자열 text_data에 대해 word cloud를 그려 변수 wordcloud에 저장
text_data = df['description'].str.cat(sep=', ')
wordcloud = wc.generate(text_data)

# - figure 함수를 사용해 시각화 공간 설정
# - figsize를 통해 공간 크기 설정
# - imshow 함수를 사용해 배열 형태의 wordcloud 시각화
# - 이미지 처리시 보간법을 bilinear로 설정
# - axis를 off로 설정해 x, y축을 삭제
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# =============================================================================
# ### 8.2.5. 특징값 추출
# =============================================================================
# #### Scikit learn에서 제공하는 TfidfVectorizer 모듈을 사용해 텍스트 데이터의 TF-IDF 값으로 문서 단어 행렬을 구성
from sklearn.feature_extraction.text import TfidfVectorizer
# - 텍스트 데이터를 astype 함수를 사용해 문자열로, tolist 함수를 사용해 리스트로 변환 후 text_data_list 변수에 저장
# - Numpy의 array 함수와 for문을 사용해 배열로 변환 후 text_data_arr에 저장
text_data_list = df['description'].astype(str).tolist()
text_data_arr = np.array([''.join(text) for text in text_data_list])

# - TfidfVectorizer 함수를 사용해 tf-idf 문서 단어 행렬을 만드는 모듈 vectorizer 선언
# - Min_df=2 : 단어 최소 등장 빈도로 2번 이상 등장하는 단어들을 대상으로 함
# - Ngram_range : 단어 추출 단위로 (1,2)는 unigram과 bigram 추출, 1<= n <= 2
# - Strip_accents=‘Unicode’ : accents(억양표시)를 제거하며 unicode에 해당하는 모든 문자에 적용 가능
# - Norm=‘l2’ : pearson 함수를 사용해 normalization 진행
# - 문서단어행렬을 만드는 모듈 vectorizer의 Fit_transform 함수를 사용해 배열에 저장된 데이터의  문서단어행렬을 구하고 matrix 형식 변수 text_data에 저장
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), strip_accents='unicode', norm='l2')
text_data = vectorizer.fit_transform(text_data_arr)
#text_data = vectorizer.fit_transform(text_data_arr[0:2]) # for test...
#text_data_arr[0:2]
# - 문서 단어 행렬을 입력한 데이터 프레임 df_tfidf
# - 행 : 문서 번호
# - 열 : token
df_tfidf = pd.DataFrame(text_data.A, columns=vectorizer.get_feature_names())
df_tfidf

# =============================================================================
# ### 8.2.6. 뉴스 분류
# =============================================================================
# #### 성능 측정 패키지 import
# - Confusion matrix : 분류 결과 건수를 나타내는 confusion matrix를 구성하는 모듈
# - Classification report : recall, precision, f-measure를 제공하는 모듈
# - f1_score : f-measure를 계산
# - Accuracy score : 정확도 수치 계산
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# #### 데이터셋 준비
# - 뉴스 내용 (description)과 카테고리 (category)를 각각 리스트형 변수 description, category로 변환
description = df['description'].astype(str).tolist()
category = df['category'].astype(str).tolist()

# #### 데이터 셋 분할
# - 데이터 셋의 80%는 training set으로, 나머지 20%는 test set으로 구성
# - 기계학습 알고리즘에 적용하기 위해서 np.array 함수를 사용해 배열로 변형
# - 뉴스 내용 description의 training set 범위 내 내용 data를 join 함수를 사용해 연결하고 x_train에 저장
# - 뉴스 카테고리 category 데이터를 y_train에 저장
trainset_size = int(round(len(description)*0.80))
x_train = np.array([''.join(data) for data in description[0:trainset_size]])
y_train = np.array([data for data in category[0:trainset_size]])
x_test = np.array([''.join(data) for data in description[trainset_size+1:len(description)]])
y_test = np.array([data for data in category[trainset_size+1:len(category)]])

# - Fit transform 함수를 사용해 training set을 기반으로 문서단어행렬 구성
# - Transform 함수를 사용해 앞 행렬을 구성한 단어들을 기반으로 문서단어행렬 구성
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

# - 각 분류 모델의 성능을 기록할 데이터 프레임 df_per 선언
df_per = pd.DataFrame(columns=['Classifier', 'F-Measure', 'Accuracy'])
df_per

# =============================================================================
# Naive Bayes
# =============================================================================
# - MultinomialNB 패키지 import
# - MultinomialNB 모듈을 사용해 naïve bayes 알고리즘으로 모델을 생성
# - Fit 함수를 사용해 모델 nb_classifier 훈련
# - Predict 함수를 사용해 test set에 대한 분류, 예측 값을 구한 후 변수 nb_pred에 저장
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB().fit(X_train, y_train)
nb_pred = nb_classifier.predict(X_test)

# - 실제 값 y_test와 예측값 nb_pred를 비교해 confusion matrix, classification report 출력
print('\n Confusion Matrix \n')
print(confusion_matrix(y_test, nb_pred))
print('\n Classification Report \n')
print(classification_report(y_test, nb_pred))

# - f1_score 함수를 사용해 실제값과 분류 결과값을 비교해 f-measure 계산
# - average=‘weighted’를 사용해 각 클래스마다 가중치 적용
# - round 함수를 사용해 소수점 2번째 자리까지 반올림
# - accuracy_score 함수를 사용해 실제값과 분류 결과값을 비교해 f-measure 계산
# - normalize=True를 통해 정확도 출력, False일 경우 올바르게 분류된 데이터 건수 출력
# - round 함수를 사용해 소수점 2번째 자리까지 반올림
# - loc 함수를 사용해 데이터 프레임에 인덱스를 지정해 입력
fm = round(f1_score(y_test, nb_pred, average='weighted'), 2)
ac = round(accuracy_score(y_test, nb_pred, normalize=True), 2)
df_per.loc[len(df_per)] = ['Naive Bayes', fm, ac]
df_per

# =============================================================================
# Decision Tree
# =============================================================================
# - Decision Tree 패키지 import
# - DecisionTreeClassifier 모듈을 사용해 decision tree 알고리즘으로 모델을 생성
# - Fit 함수를 사용해 모델 dt_classifier 훈련
# - Predict 함수를 사용해 test set에 대한 분류, 예측 값을 구한 후 변수 dt_pred에 저장
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier().fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)

# - 실제 값 y_test와 예측값 dt_pred를 비교해 confusion matrix, classification report 출력
print('\n Confusion Matrix \n')
print(confusion_matrix(y_test, dt_pred))
print('\n Classification Report \n')
print(classification_report(y_test, dt_pred))

# - f1_score 함수를 사용해 실제값과 분류 결과값을 비교해 f-measure 계산
# - average=‘weighted’를 사용해 각 클래스마다 가중치 적용
# - round 함수를 사용해 소수점 2번째 자리까지 반올림
# - accuracy_score 함수를 사용해 실제값과 분류 결과값을 비교해 f-measure 계산
# - normalize=True를 통해 정확도 출력, False일 경우 올바르게 분류된 데이터 건수 출력
# - round 함수를 사용해 소수점 2번째 자리까지 반올림
# - loc 함수를 사용해 데이터 프레임에 인덱스를 지정해 입력
fm = round(f1_score(y_test, dt_pred, average='weighted'), 2)
ac = round(accuracy_score(y_test, dt_pred, normalize=True), 2)
df_per.loc[len(df_per)] = ['Decison Tree', fm, ac]
df_per

# =============================================================================
# Random Forest
# =============================================================================
# - Random Forest 패키지 import
# - RandomForestClassifier 모듈을 사용해 random forest 알고리즘으로 모델을 생성
# - Fit 함수를 사용해 모델 rf_classifier 훈련
# - Predict 함수를 사용해 test set에 대한 분류, 예측 값을 구한 후 변수 rf_pred에 저장
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)

# - 실제 값 y_test와 예측값 rf_pred를 비교해 confusion matrix, classification report 출력
print('\n Confusion Matrix \n')
print(confusion_matrix(y_test, rf_pred))
print('\n Classification Report \n')
print(classification_report(y_test, rf_pred))

# - f1_score 함수를 사용해 실제값과 분류 결과값을 비교해 f-measure 계산
# - average=‘weighted’를 사용해 각 클래스마다 가중치 적용
# - round 함수를 사용해 소수점 2번째 자리까지 반올림
# - accuracy_score 함수를 사용해 실제값과 분류 결과값을 비교해 f-measure 계산
# - normalize=True를 통해 정확도 출력, False일 경우 올바르게 분류된 데이터 건수 출력
# - round 함수를 사용해 소수점 2번째 자리까지 반올림
# - loc 함수를 사용해 데이터 프레임에 인덱스를 지정해 입력
fm = round(f1_score(y_test, rf_pred, average='weighted'), 2)
ac = round(accuracy_score(y_test, rf_pred, normalize=True), 2)
df_per.loc[len(df_per)] = ['Random Forest', fm, ac]
df_per

# =============================================================================
# Support Vector Machine
# =============================================================================
# - SVM 패키지 import
# - LinearSVC 모듈을 사용해 SVM 알고리즘으로 모델을 생성
# Fit 함수를 사용해 모델 svm_classifier 훈련
# - Predict 함수를 사용해 test set에 대한 분류, 예측 값을 구한 후 변수 svm_pred에 저장
from sklearn.svm import LinearSVC
svm_classifier = LinearSVC().fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)

# - 실제 값 y_test와 예측값 svm_pred를 비교해 confusion matrix, classification report 출력
print('\n Confusion Matrix \n')
print(confusion_matrix(y_test, svm_pred))
print('\n Classification Report \n')
print(classification_report(y_test, svm_pred))

# - f1_score 함수를 사용해 실제값과 분류 결과값을 비교해 f-measure 계산
# - average=‘weighted’를 사용해 각 클래스마다 가중치 적용
# - round 함수를 사용해 소수점 2번째 자리까지 반올림
# - accuracy_score 함수를 사용해 실제값과 분류 결과값을 비교해 f-measure 계산
# - normalize=True를 통해 정확도 출력, False일 경우 올바르게 분류된 데이터 건수 출력
# - round 함수를 사용해 소수점 2번째 자리까지 반올림
# - loc 함수를 사용해 데이터 프레임에 인덱스를 지정해 입력
fm = round(f1_score(y_test, svm_pred, average='weighted'), 2)
ac = round(accuracy_score(y_test, svm_pred, normalize=True), 2)
df_per.loc[len(df_per)] = ['Support Vector Machine', fm, ac]
df_per

# ##### 성능 비교
# - 시각화를 위해 분류기 명을 set_index 함수를 사용해 index로 설정
df_per_1 = df_per.set_index('Classifier')
df_per_1

# - F-measure과 Accuracy 값을 plot 함수를 사용해 시각화
# - ind=‘bar’ : 막대 그래프
# - title=‘preformance’ : 그래프 제목
# - figsize : 그래프 크기 지정
# - legend : 데이터 설명
# - fontsize : 글씨 크기 
# - 그래프의 x축을 분류기 명으로 지정
# - 그래프 그리기
ax = df_per_1[['F-Measure','Accuracy']].plot(kind='bar', title ='Performance'
         , figsize=(10, 7), legend=True, fontsize=12)
ax.set_xlabel('Classifier', fontsize=12)
plt.show()
