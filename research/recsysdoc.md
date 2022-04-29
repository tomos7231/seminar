---
marp: true
theme: lab_default
paginate: true
math: true
style: | 
    :root {
        --color-background: #fff;
        --color-foreground: #333;
        --color-highlight: #356D64;
        --color-dimmed: #333;
    }
    table {
    border-spacing: 0;
    border-collapse: collapse;
    margin: 1em 0 0; 
    }
    table th,
    table td {
        padding: 0.2em 0.4em;
        border-width: 1px;
        border-style: solid; }
    th {
    text-align: center;
    color: #fff;
    background-color: #333;
    }
    tr {
    text-align: right;
    }

    section.split table, th, td {
      font-size: 14pt;
    }
    section.split table {
      width: 80%;
    }
    section.split table th {
      white-space: nowrap;
      width: 80%;
    }
---
<!-- _class: lead -->
# 推薦システムのアルゴリズム
神嶌 敏弘
研究ゼミ第2回（2022/5/9）
B4 柳智也

---
## __目次__

1. 推薦システムとは

    1.1. 推薦システムの目的

    1.2. 個人化の度合いによる推薦の種類
2. 嗜好の予測

    2.1. 内容ベースフィルタリング

    2.2.  協調フィルタリング

    2.3.  両者の比較
3. 協調フィルタリングのアルゴリズム

    3.1.  メモリベース法

    3.2.  モデルベース法

---
<!--
_footer: 1.1. 推薦システムの目的
-->

# __推薦システムとは何か__
Konstan [1]による定義：
*__Recommenders: Tools to help identify worthwhile stuff__*

- 利用者にとって有用と思われる対象、情報などを選び、利用者の目的に合わせた形で推薦するシステム

### なぜ推薦システムが必要になってきたのか？

1. 大量の情報が発信されることになった

2. 大量の情報の蓄積・流通が可能になったことで、誰もが大量の情報を得ることができるようになった

→大量に情報を受け取ることができる一方で、欲しい情報を特定できない<font color='f30100'>情報過多</font>が生じるようになった

---
<!--
_footer: 1.2. 個人化の度合いによる推薦の種類
-->

# __推薦システムの種類__
個人科の度合いによって以下の3段階に分けられる。

1. 非個人化

   全ての利用者に対して、全く同じ推薦をする。「売上ランキング」などが該当。

2. 一時的個人化

    システムを利用する一つのセッションで同じ入力や振る舞いをした利用者には同じ推薦をする。

3. 永続的個人化

    同じ入力や行動をしている利用者でも、個人情報や過去の利用履歴に応じて異なる推薦をする。

---
<!--
_footer: 1.2. 個人化の度合いによる推薦の種類
-->

# __前ページの例__
<div class="grid grid-cols-3 grid-rows-2">

<div>
</div>

<div>
</div>

<div>
</div>

<div>
<img width=400 height=250 src="https://user-images.githubusercontent.com/82197529/165898872-b2af4836-de3a-4020-9c83-8e93a4141a04.png">
</div>

<div>
<img width=400 height=250 src="https://user-images.githubusercontent.com/82197529/165899232-b461b043-7480-4863-8519-58800848a32d.png">
</div>

<div>
<img wigth=400 height=250 src="https://user-images.githubusercontent.com/82197529/165899327-9b4d7bd0-23b6-4b92-9ee4-0e6167d17ba3.png">
</div>

---
<!--
_footer: 2.嗜好の予測：内容ベースフィルタリング
-->

# __嗜好の予測__
推薦システムで最も重要な嗜好の予測の実現方法は「<font color='f30100'>内容ベースフィルタリング</font>」と「<font color='f30100'>協調フィルタリング</font>」の大きく2つに分類される

### 1.内容ベースフィルタリング

- アイテムの性質と利用者の思考パターンを比較
- アイテムの性質は、特徴ベクトルによって表される


---
<!--
_header: 1.1 線形計画モデル
-->

## 1.1.1 生産計画問題

変数に関する 1 次の等式や不等式で与えられた制約条件のもとで
変数の 1 次関数を最大化 (または最小化)する問題を__線形計画問題__ という

---
<!--
_header: 1.1 線形計画モデル
-->

## 1.1.1 生産計画問題
問題を簡潔に表現するため, $\bm{x}, \bm{A}, \bm{b}, \bm{c}$ を以下で定義すると, 

<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [["$","$"], ["\\(","\\)"]],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script>

<div class="grid grid-cols-2">

<div>

$$
\begin{aligned}
\bm{x} &= \begin{pmatrix}x_1 \\ x_2 \\ x_3 \end{pmatrix}\\
\bm{A} &= \begin{pmatrix} 
            5 & 0 & 6\\
            0 & 2 & 8\\
            7 & 0 & 15\\
            3 & 11 & 0
        \end{pmatrix}
\end{aligned}
$$

</div>

<div>

$$
\begin{aligned}
\bm{b} &= \begin{pmatrix}80\\ 50\\ 100\\ 70 \end{pmatrix}\\
\bm{c} &= \begin{pmatrix}70\\ 120\\ 30 \end{pmatrix}
\end{aligned}
$$

</div>

</div>


------
<!--
_header: 1.1 線形計画モデル
-->

## 1.1.1 生産計画問題

<div class="grid grid-rows-3">

<div>

生産計画問題は次のように書ける.

</div>

<div class="row-start-2 row-span-2">

$$
\begin{aligned}
\text{目的関数: } 
    &\bm{c}^T \bm{x} \rightarrow \text{最大}\\
\text{制約条件: } 
    &\bm{A}\bm{x} \leqq \bm{b}\\
    &\bm{x} \geqq \bm{0}
\end{aligned}
$$

</div>

</div>

---
<!--
_header: 1.1 線形計画モデル
-->
## 1.1.2 多期間計画問題

- 原料 A, B を加工して, 製品 Ⅰ, Ⅱ を製造している工場が向こう 3 カ月の生産計画を立てようとしている.
- 各月に余った製品は次月に持ち越される.

---
<!--
_header: 1.1 線形計画モデル
-->
## 1.1.2 多期間計画問題

- 原料 A, B を加工して, 製品 Ⅰ, Ⅱ を製造している工場が向こう 3 カ月の生産計画を立てようとしている.
- 各原料の利用可能量は月毎に決まっている.
- 各製品の出荷量は月毎に決まっている.
- 各製品の生産コストと在庫コストは各月を通して一定である.
- 3 カ月間にかかるコストを最小にするには月毎に各製品を何単位ずつ生産すればよいだろうか?

---
<!--
_header: 1.1 線形計画モデル
-->

<style>
table, th, td {
    font-size: 12pt;
}
</style>

## 1.1.2 多期間計画問題

<div class="grid grid-cols-3">

<div>

### 1 月

|        | Ⅰ   | Ⅱ   | 利用可能量 |
| ------ | --- | --- | ---------- |
| A      | 2   | 7   | 920        |
| B      | 5   | 3   | 790        |
| 出荷量 | 30  | 20  |            |

- hogehoge
- あああああ

</div>

<div>

### 2 月 
|        | Ⅰ   | Ⅱ   | 利用可能量 |
| ------ | --- | --- | ---------- |
| A      | 2   | 7   | 750        |
| B      | 5   | 3   | 600        |
| 出荷量 | 60  | 50  |            |

</div>

<div>

### 3 月
|        | Ⅰ   | Ⅱ   | 利用可能量 |
| ------ | --- | --- | ---------- |
| A      | 2   | 7   | 500        |
| B      | 5   | 3   | 480        |
| 出荷量 | 80  | 90  |            |

</div>

</div>

---
<!--
_header: 1.1 線形計画モデル
-->
## 1.1.3 輸送問題

---
<!--
_header: 1.2 ネットワーク計画モデル
-->
## 1.2.1 グラフとネットワーク

---
<!--
_header: 1.2 ネットワーク計画モデル
-->
## 1.2.2 最短路問題

---
<!--
_header: 1.2 ネットワーク計画モデル
-->
## 1.2.3 最大流問題と最小費用流問題

---
<!--
_header: 1.3 非線形計画モデル
-->
## 1.3.1 資源配分問題

---
<!--
_header: 1.3 非線形計画モデル
-->
## 1.3.2 ポートフォリオ選択問題

---
<!--
_header: 1.3 非線形計画モデル
-->
## 1.3.3 交通流割当問題

---
<!--
_header: 1.4 組み合わせ計画モデル
-->
## 1.4.1 生産計画問題

---
<!--
_header: 1.5 組み合わせ計画モデル
-->
## 1.4.2 固定費付き輸送問題

---
<!--
_header: 1.5 組み合わせ計画モデル
-->
## 1.4.3 ナップサック問題

---
<!--
_header: 1.5 組み合わせ計画モデル
-->

<style>
table {
    margin-left: auto;
    margin-right: auto;
    width: 50%;
    height: 40%;
}
</style>

## 1.4.4 0-1 変数についての補足




---
<!--
_header: 1.5 数理計画問題
-->
## 1.5 数理計画問題

---
## 参考文献
- 福島雅夫. 新版 数理計画入門. 第 7 版. 朝倉書店, 2017, 203p.
- 筧三郎. 工科系 線形代数 [新訂版]. 新訂版. 数理工学社, 2018, 221p.

---

## Appendix

- 線形写像
- 写像
- 線形空間

---
<!--
_header: Appendix
-->
## __線形写像__ ってなんだっけ

- 線形空間 $\bm{V}$ から 線形空間 $\bm{W}$ への写像 $f$ が以下の性質を持つ時, $f$ は __線形写像__ という.
1. $f(\bm{x_1}+\bm{x_2}) = f(\bm{x_1}) + f(\bm{x_2})$
2. $f(k\bm{x}) = kf(\bm{x})$

- 特に $\bm{V} = \bm{W}$ の時, $f$を __線形変換__ または __1次変換__ という.

---
<!--
_header: Appendix
-->
## __写像__ ってなんすか:confused:

---
## __線形空間__ ってなんだっけ

---
