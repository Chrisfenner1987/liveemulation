# FilmGrain OFX - macOS (Apple Silicon + Intel universal) build.
# Copyright (c) 2026 Chris Fenner. SPDX-License-Identifier: BSD-3-Clause

BUNDLE       = FilmGrain.ofx.bundle
PLUGIN       = FilmGrain.ofx
BUNDLE_DIR   = $(BUNDLE)/Contents/MacOS
INSTALL_DIR  = /Library/OFX/Plugins

OFX_ROOT     = openfx
OFX_INC      = $(OFX_ROOT)/include
OFXS_INC     = $(OFX_ROOT)/Support/include
OFXS_PLUG    = $(OFX_ROOT)/Support/Plugins/include
OFXS_LIB     = $(OFX_ROOT)/Support/Library

ARCH_FLAGS   = -arch arm64 -arch x86_64
# OFX_SUPPORTS_OPENGLRENDER must be defined everywhere: it gates the Metal
# command-queue plumbing in the Support library and changes struct layouts.
CXXFLAGS     = --std=c++20 -O3 -fvisibility=hidden \
               -I$(OFX_INC) -I$(OFXS_INC) -I$(OFXS_PLUG) \
               -DOFX_SUPPORTS_OPENGLRENDER $(ARCH_FLAGS)
LDFLAGS      = -bundle -fvisibility=hidden $(ARCH_FLAGS) \
               -framework Metal -framework Foundation

BUILD        = build

PLUGIN_OBJS  = $(BUILD)/FilmGrain.o $(BUILD)/FilmGrainMetal.o
SUPPORT_OBJS = $(BUILD)/ofxsCore.o $(BUILD)/ofxsImageEffect.o $(BUILD)/ofxsInteract.o \
               $(BUILD)/ofxsLog.o $(BUILD)/ofxsMultiThread.o $(BUILD)/ofxsParams.o \
               $(BUILD)/ofxsProperty.o $(BUILD)/ofxsPropertyValidation.o

all: $(BUNDLE)

$(BUNDLE): $(PLUGIN_OBJS) $(SUPPORT_OBJS)
	$(CXX) $^ -o $(PLUGIN) $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp $(PLUGIN) $(BUNDLE_DIR)/
	cp Info.plist $(BUNDLE)/Contents/
	@echo "Built $(BUNDLE)"

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/FilmGrain.o: src/FilmGrain.cpp src/FilmGrain.h src/GrainModel.h | $(BUILD)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(BUILD)/FilmGrainMetal.o: src/FilmGrainMetal.mm | $(BUILD)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(BUILD)/%.o: $(OFXS_LIB)/%.cpp | $(BUILD)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

grain_test:
	$(CXX) --std=c++17 -O2 tools/grain_test.cpp -o grain_test

clean:
	rm -rf $(BUILD) $(PLUGIN) $(BUNDLE) grain_test grain_ramp.pgm

install: $(BUNDLE)
	@echo "Installing to $(INSTALL_DIR) (requires sudo)"
	sudo rm -rf $(INSTALL_DIR)/$(BUNDLE)
	sudo cp -R $(BUNDLE) $(INSTALL_DIR)/

.PHONY: all clean install grain_test
