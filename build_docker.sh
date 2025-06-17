for arch in amd64 arm64; do
  docker buildx build --platform linux/$arch -t shl2025:$arch --load .
  docker save -o shl2025-$arch.tar shl2025:$arch
done
