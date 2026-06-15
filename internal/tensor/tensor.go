// Package tensor 定义了 Minfer 的核心数据结构：多维张量（Tensor）。
//
// 张量是 LLM 推理中最基本的数据单元。一个张量可以表示：
//   - 一个标量（0维）：比如单个注意力分数
//   - 一个向量（1维）：比如词嵌入、中间层激活
//   - 一个矩阵（2维）：比如权重矩阵、Q/K/V 投影
//   - 高维张量（3+维）：比如 batch 注意力分数 (batch×head×seq_len×seq_len)
//
// 在整个 Transformer 推理过程中，所有的数据都存储在 Tensor 中，
// 所有的运算都是 Tensor 之间的操作。这个包实现了最基本的数据结构和维度操作。
//
// 注意：Minfer 中的 Tensor 只做数据存储和维度变换，
// 真正的数值计算（矩阵乘法、Softmax 等）交给 compute.Backend 接口完成。
// 这样设计的目的是为将来添加 GPU 后端时，Tensor 本身不需要改动。
package tensor

import "fmt"

// Tensor 是多维数组的数据结构。
//
// Data 按行优先（row-major）顺序存储在一维切片中。
// 对于形状 [d0, d1, d2, ..., dn] 的张量，元素 (i0, i1, ..., in) 的索引为：
//
//     offset = i0*(d1*d2*...*dn) + i1*(d2*...*dn) + ... + in
//
// 行优先存储意味着最右侧的维度变化最快（C 风格）。
//
// 例子：形状 [2, 3] 的矩阵
//   ⎡ a  b  c ⎤     在内存中: [a, b, c, d, e, f]
//   ⎣ d  e  f ⎦     (0,0)=a, (0,1)=b, (0,2)=c, (1,0)=d, (1,1)=e, (1,2)=f
type Tensor struct {
	Data  []float32 // 扁平化存储的数值数据
	Shape []int     // 各维度大小，例如 [2, 3] 表示 2 行 3 列
}

// New 创建一个新的张量并分配内存。
// shape 参数指定每个维度的大小。
//
// 例子：
//   tensor.New(2, 3)     → 形状 [2, 3]，6 个元素，初始值为 0
//   tensor.New(4)        → 形状 [4]，4 个元素（1维向量）
//   tensor.New(2, 3, 4)  → 形状 [2, 3, 4]，24 个元素（3维张量）
func New(shape ...int) *Tensor {
	// 计算总元素数：各维度乘积
	size := 1
	for _, d := range shape {
		size *= d
	}
	return &Tensor{
		Data:  make([]float32, size),
		Shape: copyShape(shape),
	}
}

// NewWithData 从现有的数据切片创建张量。
// 不复制 data，直接引用（零拷贝）。
//
// 参数：
//   - data: 已有的 []float32 数据
//   - shape: 各维度大小
//
// shape 各维度的乘积必须等于 len(data)，否则 panic。
func NewWithData(data []float32, shape ...int) *Tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	if len(data) != size {
		panic("tensor.NewWithData: data length does not match shape")
	}
	return &Tensor{
		Data:  data,
		Shape: copyShape(shape),
	}
}

// Dims 返回张量的维数。
// 标量返回 0，向量返回 1，矩阵返回 2，以此类推。
func (t *Tensor) Dims() int {
	return len(t.Shape)
}

// Size 返回第 i 维的大小。如果 i 超出范围，返回 1。
// 这样设计是为了方便处理标量或广播场景。
func (t *Tensor) Size(i int) int {
	if i < 0 || i >= len(t.Shape) {
		return 1
	}
	return t.Shape[i]
}

// NumElements 返回张量中的总元素数量（各维度乘积）。
// 等价于 len(t.Data)。
func (t *Tensor) NumElements() int {
	return len(t.Data)
}

// At 返回指定索引位置的元素值（不修改原张量）。
// 使用可变参数来适配任意维度的张量。
//
// 例子：
//   t.At(0, 1)  → 形状 [2,3] 的矩阵中第 0 行第 1 列的元素
//   t.At(3)     → 1维向量中第 3 个元素
//
// 计算偏移量的公式：
//   offset = Σ(i_n × stride_n)，其中 stride 是各维度的跨度
//
// Panic 条件：
//   - 索引数量不等于维度数（len(indices) != len(t.Shape)）
//   - 某个索引超出对应维度范围
func (t *Tensor) At(indices ...int) float32 {
	if len(indices) != len(t.Shape) {
		panic("tensor.At: index count does not match tensor dimensions")
	}
	offset := 0
	for i, idx := range indices {
		if idx < 0 || idx >= t.Shape[i] {
			panic("tensor.At: index out of range")
		}
		// stride = 当前维度之后所有维度的乘积
		stride := 1
		for j := i + 1; j < len(t.Shape); j++ {
			stride *= t.Shape[j]
		}
		offset += idx * stride
	}
	return t.Data[offset]
}

// Set 设置指定索引位置的元素值。
// 用法同 At，只是写入而不是读取。
//
// Panic 条件同 At。
func (t *Tensor) Set(val float32, indices ...int) {
	if len(indices) != len(t.Shape) {
		panic("tensor.Set: index count does not match tensor dimensions")
	}
	offset := 0
	for i, idx := range indices {
		if idx < 0 || idx >= t.Shape[i] {
			panic("tensor.Set: index out of range")
		}
		stride := 1
		for j := i + 1; j < len(t.Shape); j++ {
			stride *= t.Shape[j]
		}
		offset += idx * stride
	}
	t.Data[offset] = val
}

// View 返回一个形状不同但共享同一底层数据的张量。
// 这是零拷贝操作——只改变 Shape，不移动数据。
//
// 约束：新形状的元素总数必须等于旧形状的元素总数。
//
// 用途：在 Transformer 中频繁需要 reshape。
// 例如 attention 计算中：
//   x: [seq_len, hidden_dim]
//   需要 reshape 为 [seq_len, num_heads, head_dim] 以便按头计算
//   用 View 可以零成本做到。
//
// Panic 条件：新形状的元素总数不等于当前元素总数。
func (t *Tensor) View(shape ...int) *Tensor {
	newSize := 1
	for _, d := range shape {
		newSize *= d
	}
	oldSize := len(t.Data)
	if newSize != oldSize {
		panic(fmt.Sprintf(
			"tensor.View: new shape has %d elements, but tensor has %d elements",
			newSize, oldSize,
		))
	}
	return &Tensor{
		Data:  t.Data,
		Shape: copyShape(shape),
	}
}

// Clone 创建张量的深度拷贝。
// 返回一个全新的 Tensor，拥有独立的数据内存。
// 修改克隆体不会影响原张量。
//
// 何时需要 Clone：
//   - 在计算中需要保留中间结果的快照（比如残差连接）
//   - 需要修改数据又不想影响原始张量
func (t *Tensor) Clone() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	return &Tensor{
		Data:  data,
		Shape: copyShape(t.Shape),
	}
}

// copyShape 复制 shape 切片以防止外部修改影响张量状态。
func copyShape(s []int) []int {
	c := make([]int, len(s))
	copy(c, s)
	return c
}
