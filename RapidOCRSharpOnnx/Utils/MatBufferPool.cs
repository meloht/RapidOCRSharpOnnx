using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RapidOCRSharpOnnx.Utils
{
    public sealed class MatBufferPool : IDisposable
    {
        private int _inputSizeInBytes;
        private long[] _inputShape;
        /// <summary>
        /// 实际缓存池
        /// </summary>
        private readonly ConcurrentBag<ImageBatchData> _pool = new();

        /// <summary>
        /// 最大缓存数量（超过则直接 Dispose）
        /// </summary>
        private readonly int _poolSzie;


        /// <summary>
        /// 当前正在使用中的对象数量
        /// </summary>
        private int _usedCount;

        private bool _disposed;

        public MatBufferPool(int poolSzie, int inputSizeInBytes, long[] inputShape)
        {
            _usedCount = 0;
            _inputSizeInBytes = inputSizeInBytes;
            _inputShape = inputShape;
            _poolSzie = poolSzie;


            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(_poolSzie);
            // 预热池
            for (int i = 0; i < _poolSzie; i++)
            {
                _pool.Add(new ImageBatchData(_inputSizeInBytes, _inputShape));
            }
        }

        /// <summary>
        /// 当前使用中的对象数量
        /// </summary>
        public int UsedCount => Volatile.Read(ref _usedCount);

        public ImageBatchData Rent()
        {
            ThrowIfDisposed();

            Interlocked.Increment(ref _usedCount);

            if (_pool.TryTake(out var item))
            {
                return item;
            }

            // 池空了，临时创建
            return new ImageBatchData(_inputSizeInBytes, _inputShape);

        }
        /// <summary>
        /// 归还对象
        /// </summary>
        public void Return(ImageBatchData item)
        {
            if (item == null)
                return;

            if (_disposed)
            {
                item.Dispose();
                return;
            }

            // 如果你有 Clear() / Reset() 方法，建议这里调用
            // item.Reset();

            Interlocked.Decrement(ref _usedCount);

            // 超过池容量 -> 直接销毁
            if (_pool.Count < _poolSzie)
            {
                _pool.Add(item);
            }
            else
            {
                item.Dispose();
            }
        }


        /// <summary>
        /// 清空池
        /// </summary>
        public void Clear()
        {
            while (_pool.TryTake(out var item))
            {
                item.Dispose();
            }

        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;

            Clear();

            GC.SuppressFinalize(this);
        }
    }
}
