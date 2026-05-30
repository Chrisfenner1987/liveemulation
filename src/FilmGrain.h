// FilmGrain OFX - plugin factory.
// Copyright (c) 2026 Chris Fenner. SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ofxsImageEffect.h"

class FilmGrainFactory : public OFX::PluginFactoryHelper<FilmGrainFactory> {
public:
    FilmGrainFactory();
    virtual void load() {}
    virtual void unload() {}
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc);
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context);
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum p_Context);
};
