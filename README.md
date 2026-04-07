# FEMの基礎からPINNsまで① — 1次元棒の引っ張り問題

1次元の棒の引っ張り問題を **理論解 / 有限要素法 (FEM) / PINNs** の3手法で解いて結果を比較するコードです。

元記事: [FEMの基礎からPINNsまで① - Qiita](https://qiita.com/udi_/items/345e71aa592d1e8cada6)

---

## 問題設定

左端固定・右端に集中荷重 $P$ がかかる棒の変位分布 $u(x)$ を求めます。

| パラメータ | 記号 | 値 |
|---|---|---|
| 棒の長さ | $L$ | 3 m |
| ヤング率 | $E$ | 2 N/m² |
| 断面積 | $A$ | 1 m² |
| 集中荷重 | $P$ | 10 N |

### 理論解

$$u(x) = \frac{Px}{EA} \quad \Rightarrow \quad u(L) = \frac{PL}{EA} = 15 \text{ m}$$

---

## 手法の概要

### 有限要素法 (FEM)

棒を複数の要素に分割し、要素剛性マトリクスを組み立てて連立方程式を解きます。

$$K^e = \frac{EA}{L^e}, \quad \mathbf{K}\mathbf{u} = \mathbf{F}$$

### PINNs (Physics-Informed Neural Networks)

支配方程式・境界条件を損失関数として学習します。

- **支配方程式 (PDE):** $EA \, u''(x) = 0$
- **Dirichlet BC:** $u(0) = 0$（ネットワーク構造で保証: `u = x * net(x)`）
- **Neumann BC:** $EA \, u'(L) = P$

$$\mathcal{L} = \mathcal{L}_\text{PDE} + \lambda \, \mathcal{L}_\text{BC}$$

---

実行すると以下が出力されます:

- FEM の各節点変位
- PINNs の学習ログ (1000 epoch ごと)
- 3手法の比較グラフ (`output.png`)
- 右端変位 $u(L)$ の比較

---

## 期待される出力

```
=== 1D Bar: FEM + PINNs ===
  E=2.0, A=1.0, P=10.0, L=3.0
  理論解 u(L) = 15.0000 m

--- FEM (2 elements) ---
  x=0.00 → u=0.0000 m
  x=1.50 → u=7.5000 m
  x=3.00 → u=15.0000 m

--- u(L) 比較 ---
  理論解  : 15.0000 m
  FEM     : 15.0000 m  (n=2 elements)
  PINNs   : 15.00xx m
```

---

## 参考文献

- 元記事: [FEMの基礎からPINNsまで① - Qiita](https://qiita.com/udi_/items/345e71aa592d1e8cada6)
- Raissi et al. (2019), *Physics-informed neural networks*, Journal of Computational Physics
