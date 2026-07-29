# Insta360 Frame Extractor GUI

Windows 上で Insta360 などの equirectangular 360° 動画から、
任意方向の JPG 連番を切り出すための GUI ツールです。

`ffmpeg` の `v360` フィルタを使って視点変換を行い、必要に応じて
SegFormer ベースのマスク画像も同時生成できます。

## 主な機能

- 入力動画 / 出力フォルダの指定
- `yaw`, `pitch` ごとの複数方向書き出し
- 単一デコードでの全方向同時抽出
- RealityScan / RealityCapture 用 XMP サイドカーの書き出し
- 水平リング生成
- FPS / 画角 / 出力サイズ / 並列数の指定
- 逆再生
- 奇数フレーム時の方向 index 反転命名
- 画像抽出のみ / マスク生成のみ / 両方の切り替え
- SegFormer による空 / 人 / 車 / 木の個別マスク
- マスクの細かさプリセット、除外しきい値、マスクバッチ数
- 入力動画プレビュー
  - 高速再生
  - 枠位置優先プレビュー
  - シークバー
  - ウィンドウサイズに追従 (アスペクト比を保って自動フィット)
- 書き出し方向の 3D 球体ビュー
- 設定の自動保存 / 復元

## 動作環境

- Windows 10 / 11
- Python 3.10 以上
- `ffmpeg` / `ffprobe` が PATH に通っていること

高速プレビューを使う場合:

- VLC Desktop のインストール
  - 既定では `C:\Program Files\VideoLAN\VLC\libvlc.dll` を利用
- `python-vlc`

マスク生成を使う場合:

- `transformers`
- `torch`
  - GPU を使うなら環境に合った CUDA 対応版

## セットアップ

### 1. リポジトリを取得

```powershell
git clone https://github.com/kazu107/instaffmpeg.git
cd instaffmpeg
```

### 2. 仮想環境を作成

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. 基本依存をインストール

```powershell
python -m pip install -r requirements.txt
```

### 4. `ffmpeg` を使えるようにする

以下が通る状態にしてください。

```powershell
ffmpeg -version
ffprobe -version
```

### 5. 高速プレビューを使う場合

VLC Desktop を入れたうえで:

```powershell
python -m pip install python-vlc
```

### 6. マスク生成を使う場合

CPU のみで良ければ:

```powershell
python -m pip install transformers torch
```

GPU を使う場合は、使用環境に合った `torch` を入れてから
`transformers` を追加してください。

## 起動方法

```powershell
.\scripts\insta360-gui.cmd
```

または:

```powershell
python .\insta360_frame_extractor_gui.py
```

## 使い方

### 基本フロー

1. 入力動画を選ぶ
2. 出力フォルダを指定する
3. 書き出し方向を設定する
4. 必要なら FPS / 画角 / 解像度 / 並列数を調整する
5. 必要ならマスク設定を有効にする
6. `実行` を押す

### 書き出し方向

- `yaw`, `pitch` を直接追加 / 更新できます
- `水平リング生成` を使うと一定間隔の方向をまとめて作れます
- 方向セットはタブで複数管理できます

### プレビュー

- `枠位置優先`
  - 再生映像と書き出し方向の枠位置を一致させたい時に使います
- `枠位置優先` を OFF
  - VLC を使った高速再生を優先します
- どちらのモードでも、プレビューはウィンドウサイズに追従して
  アスペクト比を保ったまま表示領域に収まります
- 起動時のウィンドウ最小サイズは UI の実要求サイズから自動で決まります
  - これより小さくはできません (小さくすると操作ボタンが隠れてしまうため)

### 実行モード

- `画像抽出`
  - JPG を生成
- `マスク生成`
  - 既存 JPG または新規生成 JPG に対してマスクを生成

両方 ON:

- 画像抽出後にマスクを生成

`マスク生成` のみ ON:

- 出力フォルダ内の JPG に対してマスクを作成

## 出力ファイル

### 抽出画像

```text
{動画名}_{frame_index}_{direction_index}.jpg
```

例:

```text
room360_0000_00.jpg
room360_0000_01.jpg
room360_0001_07.jpg
```

- `frame_index` は 4 桁ゼロ埋め
- `direction_index` は方向数に応じてゼロ埋め

### マスク画像

```text
{元画像名}.mask.png
```

例:

```text
room360_0000_00.jpg.mask.png
```

- 白: 使用
- 黒: 除外
- 検出がなかった場合はマスク画像を作りません

### XMP サイドカー

```text
{動画名}_{frame_index}_{direction_index}.xmp
```

例:

```text
room360_0000_00.xmp
```

- `RealityScan用XMPを書き出す` が ON の時だけ生成します

## 単一デコードでの全方向抽出

`単一デコードで全方向を抽出` (既定 ON) を使うと、入力動画のデコードが
方向数と同じ回数から 1 回に減ります。

方向ごとに `ffmpeg` を起動する従来の方式では、方向数だけ入力を丸ごとデコードし直します。
デコードは `v360` の視点変換よりもはるかに重いため、ここが全体の律速になります。
このモードでは `split` で分岐して 1 回のデコードを全方向で共有します。

出力画素は従来方式とバイト単位で一致します。

実測 (7680x3840 / H.264 / 30秒 / 6方向 / fps=2 / 出力 1920x1440):

| 方式 | 所要時間 |
|---|---|
| 方向ごとに `ffmpeg` を起動 (6回デコード) | 151.9 秒 |
| 単一デコード | 31.1 秒 |

360 枚すべてがバイト単位で一致することを確認済みです。

方向数が多いほど効果が大きくなります。1 プロセスにまとまるため `並列数` の
指定は単一デコード時には使われません。

## RealityScan / RealityCapture 用 XMP

`RealityScan用XMPを書き出す` を ON にすると、各 JPG の隣に `<画像名>.xmp` を出力します。
RealityScan は画像を読み込む際にこのサイドカーを自動で拾います。

`v360` の `flat` 出力は理想ピンホール投影そのものです。

- レンズ歪みが無い
- 主点が画像中心にある
- 画素が正方 (`v_fov` を水平画角とアスペクト比から導いているため)

つまり内部パラメータは推定するまでもなく確定しています。これを事前情報として渡すことで、
RealityScan 側の探索範囲が狭まり、アライメントが速く安定します。

書き出す内容:

- `xcr:FocalLength35mm` — 水平画角から厳密に計算した 35mm 換算焦点距離
  - 長辺基準。`h_fov=90` かつ横長なら 18mm
- `xcr:DistortionModel="division"` — 係数を書かないので歪みゼロ
- `xcr:CalibrationGroup` / `xcr:DistortionGroup` — 全画像で共有
  - 「これらは全て同一の合成カメラで撮られた」と伝わり、キャリブレーション推定が 1 グループに集約されます

姿勢 (`xcr:Rotation` / `xcr:Position`) は書き出しません。
`yaw` / `pitch` は既知ですが、`xcr:Rotation` の座標系規約を実機で確認できていないため、
誤った姿勢事前情報でアライメントを壊すより、確実な内部パラメータだけを渡す方が安全です。

## よく使う設定

### 高速に抽出したい

- `単一デコードで全方向を抽出` を ON にする (既定 ON)
- GPU が使えるなら `CUDAデコードを使う` を ON
- 従来方式に戻した場合のみ `並列数` を 2〜4 から試す

### マスクを速くしたい

- `バッチ数` を 2〜4 から試す
  - 一度に推論する枚数です。指定できるのは 1〜8 です
- `細かさ = 標準`

### マスク境界を細かくしたい

- `細かさ = 高` または `最高`
- `除外しきい値` を調整

### 誤検出を減らしたい

- `除外しきい値` を上げる
  - 指定できるのは 0 より大きく 1 以下です
  - 0 を許すと全面が除外されてしまうため受け付けません
- `空 / 人 / 車 / 木` を必要な対象だけ ON にする
  - 対象は SegFormer (ADE20K) の該当ラベルのみです
    - `車` = `car` のみ。バス / トラック / バン / 自転車は除外されません
    - `木` = `tree` のみ。草 / 植物 / 花は除外されません

## 補足

- `yaw/pitch` の並び順が基本の `direction_index` になります
- `奇数フレームで方向indexを逆順` を ON にすると、
  奇数フレームのみ方向 suffix の付け方を逆順にします
- 設定は終了時に `.insta360_frame_extractor_gui.settings.json` へ保存されます

## 開発メモ

- エントリポイント: [insta360_frame_extractor_gui.py](./insta360_frame_extractor_gui.py)
- 起動ラッパー: [scripts/insta360-gui.cmd](./scripts/insta360-gui.cmd)
- テスト:

```powershell
python -m unittest discover -s tests -v
```

## 旧ファイルについて

`compose.yaml` は過去の実験用ファイルとして残っていますが、
現行の GUI 利用には不要です。
