---
marp: true
theme: gaia
footer: 2022/OO/OO
paginate: true
math: true
style: | 
    @import url('https://unpkg.com/tailwindcss@^2/dist/utilities.min.css');
    :root {
        --color-background: #fff;
        --color-foreground: #333;
        --color-highlight: #f00;
        --color-dimmed: #888;
    }
    section {
        color: black;
        font-family: メイリオ;
        justify-content: normal;
    }
    section.lead {
        justify-content: center;
    }
    section::after {
        content: attr(data-marpit-pagination)"/"attr(data-marpit-pagination-total);
        font-size: 60%;
        color: gray;
    }
    footer {
        text-align: left;
        font-size: 60%;
        color: gray;
    }
    header {
        text-align: center;
        font-size: 60%;
        color: gray;
    }
    .columns {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
    }
---

<!-- _class: lead -->

# 多変量解析入門
### 永田 靖・棟近雅彦
輪読ゼミ第２回（2022/4/27）
教科書：p00〜p00
担当者：B4 柳 智也


---
## 単回帰分析

#### **単回帰分析**とは？
→ある説明変数 𝑥 から目的変数 𝑦 を制御・予測すること

**解析ストーリー**
①最小二乗法による回帰式の推定
②寄与率・自由度調整済寄与率による回帰式の性能評価
③回帰係数の検定・区間推定
④残差・テコ比を用いた回帰式の妥当性の検討
⑤得られた回帰式による予測


---
<!--
_header: 1.1 線形計画モデル
-->

## ①最小二乗法による回帰式の推定
以下の単回帰モデルを想定
$$
y_i = β_0 + β_1x_i + ε_i, ε_i～N(0, σ^{2})
$$

---
<!--
_header: 1.1 線形計画モデル
-->

## 1.1.1 生産計画問題
製品 Ⅰ, Ⅱ, Ⅲ の生産量を $x_1, x_2, x_3$ 単位とする.

|      | Ⅰ | Ⅱ  | Ⅲ | 利用可能量 (単位) |
| :---: | ---: | ---: |---: | ---: |
| A | 5 | 0   | 6 | 80 |
| B | 0 | 2   | 8 | 50 |
| C | 7 | 0   | 15 | 100 |
| D | 3 | 11   | 0 | 70 |
| 利益 (万円) | 70 | 120 | 30 |  |
---
<!--
_header: 1.1 線形計画モデル
-->

## 1.1.1 生産計画問題

生産計画問題は以下の制約条件のもとで目的関数が最大になるような変数 $x_1, x_2, x_3$ を求める問題に定式化できる.

$$

\begin{equation}
\int^{b}_{a} f(x) dx = \lim_{n \to \infty} \sum^{n-1}_{i=1} f(x_{i}) \Delta x
\end{equation}

\begin{aligned}
\verb|目的関数: | 
    &70x_1 + 120x_2 + 30x_3 \verb| (万円)|\\
\verb|制約式: | 
    &5x_1 + 6x_3 \leqq 80 \\
    &2x_2 + 8x_3 \leqq 50 \\
    &7x_1 + 15x_3 \leqq 100\\
    &3x_1 + 11x_2 \leqq 70\\
    &x_1, x_2, x_3 \geqq 0
\end{aligned}
$$

---
<!--
_header: 1.1 線形計画モデル
-->

## 1.1.1 生産計画問題

変数に関する 1 次の等式や不等式で与えられた制約条件のもとで
変数の 1 次関数を最大化 (または最小化)する問題を

### __線形計画問題__ という

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
\verb|目的関数: | 
    &\bm{c}^T \bm{x} \rightarrow \verb|最大|\\
\verb|制約式: | 
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
- 各原料の利用可能量は月毎
- 各製品の出荷量は月毎に決まっている.

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
