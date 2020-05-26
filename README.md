# BL-SOMの学習の仕方
---
[BL-SOM](../../pdf_source/BL-SOM.pdf)はアルゴリズムの性質上、主成分分析を必要とします。<br>
ですので、R言語を利用して、[機械学習用データセット作成ツール](../機械学習用データセット作成ツール/ReadMe.md)で作成したSOM学習用データセットを用いて<br>主成分分析を行う必要があります。<font color="red">また、その際にRを用いて以下の3つのデータセットを作成する必要があります。</font>

- vector_Ave.txt　.......　入力平均ベクトルが保存されたファイル
- rotation.txt 　　.......　第1主成分、第2主成分の固有ベクトル
- sdev.txt　　 　 .......　第1主成分、第2主成分の標準偏差

作成した3つのファイルは，[機械学習用データセット作成ツール](../機械学習用データセット作成ツール/ReadMe.md)で作成される PCA　又は、PCA01 フォルダの中に保存してください。

## 1.　AP情報の取得
DataBaseMakerを使って、位置推定を行いたい環境で測定を行ってください。

## 2.　機械学習用データセット作成ツールを使用
BataBaseMakerで保存されたファイル群に対して<br>[機械学習用データセット作成ツールを使って、SOM学習用データセットを作成してください。](../機械学習用データセット作成ツール/ReadMe.md)

## 3.　R言語で主成分分析を行う
2で作成した pca_target.csv 又は、pca_01_target.csv を用いて、主成分分析を行う。
以下に手順を示す。<br>
##### 1.　対象となるpca_target.csvのファイルパスを指定し、読み込みを行う。
~~~
pca<-read.csv("pca_target.csv",header=TRUE)
~~~

##### 2.　1列目のラベルは不要なので2列目以降を抜き出す。
~~~
pca<-pca[2:ncol(pca)]
~~~
ラベルの確認は、以下のコマンドなどで確認する。
~~~
fix(pca)
~~~

##### 3.　prcompで主成分分析を行う。第2引数をTで相関行列を、Fで分散共分散行列を利用して主成分分析を行うので、任意で設定する。
~~~
resultF<-prcomp(pca,scale=F)
resultT<-prcomp(pca,scale=T)
~~~

##### 4.　第1主成分、第2主成分の固有ベクトルを<font color="red">rotation.txt</font>の名前で保存
固有ベクトルを抜き出して
~~~
rotation <- resultF$rotation[,1:2]
		 	or
rotation <- resultT$rotation[,1:2]
~~~
任意のファイルパスに保存する
~~~
write.table(rotation,"PCA\\rotation.txt",quote=F)
~~~

##### 5.　第1主成分、第2主成分の標準偏差を<font color="red">sdev.txt</font>の名前で保存
標準偏差を抜き出して
~~~
sdev<-resultF$sdev[1:2]
		 or
sdev<-resultT$sdev[1:2]
~~~

任意のファイルパスに保存する
~~~
write.table(sdev,"PCA\\sdev.txt",quote=F)
~~~

##### 6.　平均ベクトルを<font color="red">vector_Ave.txt</font>の名前で保存
行列の次元毎の平均値を計算して保存
~~~
aveVec<-colMeans(pca)
write.table(aveVec,"PCA\\vector_Ave.txt",quote=F)
~~~

##### 7.　BLSOM_GPUフォルダ内の、BLSOM.slnを起動し、実行する
<font color=red>※BLSOM.cpp, BLSOM.h はEigenライブラリを使用します。インストールするか、直下に保存してincludeしておいてください。</font>
BLSOM.cpp、BLSOM.h、position.h　をincludeすれば、お好きなプロジェクトで利用できますが、<font color=red>二次配布は禁止です。(誰もしないと思うけど…) </font>
<p>また、BLSOM.cppでは、cudaではなく、ampを利用している為、実行速度がcuda比べて若干遅い。設計がampの制約に縛られている。等の問題点があります。
<p>もし、不明なエラーが出た場合は、GPUの性能不足である可能性があります(1処理が2000msを超えるとカーネルが強制終了する等の縛りなどがある)

以下に、注意すべき関数のみ説明を行う。
main.cppの36行目に利用されている **averageClassication(std::string filepath)**について

~~~
batchSom.averageClassification("任意のファイルパス\\No.1.epc"); //実行時、average_epoc.txtが作成される。
~~~

では、U-matrixで得られるイメージ画像に割り振られる座標に割り振られた連番に対応するベクトルデータを選択する。
<p>サンプルでは、平均ベクトルを作るのが面倒なので、Np.1.epcをU-matrix上に割り振っている。
この時、対象となるファイルは、epc形式と同じフォーマットで作成する必要がある。(ラベル、ヘッダが必要。区切り文字は \t )

##### 8.　U-matrixフォルダ内の、u_matrix.pyを起動
~~~
batchSom.averageClassification("任意のファイルパス\\No.1.epc");
~~~
で、得られたaverage_epoc.txtと
~~~
	std::string mapName[] = { "HimejiGF","UnivHyogo2_4","UnivHyogo2_4_01" };
	std::string outputName = mapName[1] + "_iA_" + std::to_string(initAlfa) +"_iB_" + std::to_string(initBeta) +"_timeA_" + std::to_string(timeAlfa) +"_timeB_" + std::to_string(timeBeta) +"_Learn" + std::to_string(learn_num) +".umat";
	
	std::cout << "Output Umatrix" << std::endl;
	std::ofstream outputfile(outputName);
	for (int i = 0; i < batchSom.MapHeight(); i++){
		for (int j = 0; j < batchSom.MapWidth(); j++){
			outputfile << umat[i][j] << "\t";
		}
		outputfile.seekp(-1, std::ofstream::ios_base::end);
		outputfile << "\n";
	}
	outputfile.seekp(-1, std::ofstream::ios_base::end);
	outputfile << "\0";
	outputfile.close();
	std::cout << "Finish Umatrix" << std::endl;
~~~
で、得られたumat形式のファイルを利用して、[U-matrixを実行する。](../U-matrix/ReadMe.md)

##### 9.　クラスタリングを可視化する
[推定エリア色分けツール](../推定エリア色分けツール/ReadMe.md)を利用して、クラスタリング結果を可視化する。
