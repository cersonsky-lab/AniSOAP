# This is only temporary -- presumably, this compilation & moving resulting
# shared object / dynamically linked libraries to appropriate folder should
# be taken care of with setup.py
# Delete original file

ifeq ($(OS), Windows_NT)
	CURR_OS := windows
	EXT := dll
	MOVE_EXT := pyd
else
	ifeq ($(shell uname), Darwin)
		CURR_OS := mac
		EXT := .dylib
		MOVE_EXT := so
	else
		CURR_OS := linux
		EXT := so
		MOVE_EXT := so
	endif
endif

run: compile_$(CURR_OS) move_$(CURR_OS)

compile_windows: compile

compile_linux: compile

compile_mac:
	cargo rustc --release -- -C link-arg=-undefined -C link-arg=dynamic_lookup

compile: src/lib.rs Cargo.toml
	cargo build --release

move_windows:
	echo F|xcopy /y "target\release\anisoap_rust_lib.$(EXT)" "..\anisoap\lib\anisoap_rust_lib.$(MOVE_EXT)"

move_linux:
	cp -f target/release/libanisoap_rust_lib.$(EXT) ../anisoap/lib/anisoap_rust_lib.$(MOVE_EXT)

move_mac:
	cp -f target/release/libanisoap_rust_lib.$(EXT) ../anisoap/lib/anisoap_rust_lib.$(MOVE_EXT)

# # Build the library and move to the correct folder.
# linux: build move_linux

# # TODO: Further testing is necessary to confirm that windows version works correctly.
# windows: build move_windows

# # TODO: Further testing is necessary to confirm that mac version works correctly.
# mac: build move_mac

# # Command to build shared object (.so) or dynamically linked libraries (.dll)
# build:
# 	cargo build --release

# # Move the build .so file to correct spot with correct name
# move_linux:
# 	rm -f ../anisoap/lib/anisoap_rust_lib.so
# 	cp target/release/libanisoap_rust_lib.so ../anisoap/lib/anisoap_rust_lib.so

# # TODO: Further testing is necessary to see if move command actually works.
# move_windows:
# 	rm -f ../anisoap/lib/anisoap_rust_lib.pyd
# 	cp target/release/libanisoap_rust_lib.so ../anisoap/lib/anisoap_rust_lib.pyd

# # TODO: Further testing is necessary to see if move command actually works.
# move_mac:
# 	rm -f ../anisoap/lib/anisoap_rust_lib.dylib
# 	cp target/release/libanisoap_rust_lib.so ../anisoap/lib/anisoap_rust_lib.dylib