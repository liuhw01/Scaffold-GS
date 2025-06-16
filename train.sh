function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

# 调用 rand 函数，在 10000~30000 之间生成一个端口号（避免端口冲突）。
port=$(rand 10000 30000)


# 默认的 level-of-detail 为 0；
# 训练迭代次数为 30,000；
# 默认不启用 warmup 预热。
lod=0
iterations=30_000
warmup="False"

# 支持传入的参数有：

# --logdir: 输出目录名（日志等）

# --data: 数据集名称（或路径下的子目录）

# --lod: LOD 级别（默认 0）

# --gpu: GPU ID

# --warmup: 是否开启 warmup（True / False）

# --voxel_size: 体素大小

# --update_init_factor: 锚点更新因子

# --appearance_dim: 外观特征维度

# --ratio: 某个控制比例（比如 anchor 数量占比）

# 若遇到未知参数，则退出。
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        --lod) lod="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --warmup) warmup="$2"; shift ;;
        --voxel_size) vsize="$2"; shift ;;
        --update_init_factor) update_init_factor="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

if [ "$warmup" = "True" ]; then
    python train.py --eval -s data/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
else
    python train.py --eval -s data/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
fi
