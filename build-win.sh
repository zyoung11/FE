#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MINGW_ROOT="${MINGW_ROOT:-$HOME/.local/share/mingw-w64}"
MINGW_LIB="$MINGW_ROOT/usr/x86_64-w64-mingw32/lib"
MINGW_INC="$MINGW_ROOT/usr/x86_64-w64-mingw32/include"
ODIN_ROOT="$(odin root)"
WINBUILD=/tmp/retraced-winbuild

if [ ! -d "$MINGW_LIB" ]; then
    echo "[-] mingw-w64 库未找到: $MINGW_LIB"
    echo "    从 https://archlinux.cachyos.org/repo/extra/os/x86_64/ 下载并解压:"
    echo "    mingw-w64-crt / mingw-w64-headers 到 \$HOME/.local/share/mingw-w64"
    exit 1
fi

if command -v nvcc >/dev/null 2>&1; then
    nvcc -ptx -arch=compute_80 -O3 -use_fast_math kernel.cu -o kernel.ptx
fi

rm -rf "$WINBUILD"
mkdir -p "$WINBUILD"

clang --target=x86_64-w64-windows-gnu -O2 -DNDEBUG -c \
    "$ODIN_ROOT/vendor/stb/src/stb_image.c" -I"$MINGW_INC" -o "$WINBUILD/stb_image.o"
clang --target=x86_64-w64-windows-gnu -O2 -DNDEBUG -c \
    "$ODIN_ROOT/vendor/stb/src/stb_image_write.c" -I"$MINGW_INC" -o "$WINBUILD/stb_image_write.o"
clang --target=x86_64-w64-windows-gnu -O2 -DNDEBUG -c \
    "$ODIN_ROOT/vendor/stb/src/stb_image_resize.c" -I"$MINGW_INC" -o "$WINBUILD/stb_image_resize.o"
clang --target=x86_64-w64-windows-gnu -c build/win/crt_shim.asm -o "$WINBUILD/crt_shim.obj"
clang --target=x86_64-w64-windows-gnu -c build/win/tls_shim.asm -o "$WINBUILD/tls_shim.obj"
clang --target=x86_64-w64-windows-gnu -c build/win/fopen_shim.c -o "$WINBUILD/fopen_shim.obj"

odin build . -target:windows_amd64 -build-mode:obj -o:speed
trap 'rm -f *.obj' EXIT

lld-link -lldmingw /entry:main /subsystem:console /out:retraced-odin.exe \
    *.obj \
    "$WINBUILD/stb_image.o" "$WINBUILD/stb_image_write.o" "$WINBUILD/stb_image_resize.o" \
    "$WINBUILD/crt_shim.obj" "$WINBUILD/fopen_shim.obj" "$WINBUILD/tls_shim.obj" \
    /libpath:"$MINGW_LIB" \
    /nodefaultlib:LIBCMT /nodefaultlib:OLDNAMES \
    /defaultlib:msvcrt /defaultlib:kernel32 /defaultlib:bcrypt \
    /defaultlib:advapi32 /defaultlib:ws2_32

echo "[+] retraced-odin.exe (Windows x86_64)"
