//==============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "SetBuilderOptions.hpp"
#include "DlSystem/TensorShapeMap.hpp"

#include "SNPE/SNPE.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::UDLBundle udlBundle,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessor(runtime)
       .setUdlBundle(udlBundle)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .build();
    return snpe;
}

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::UDLBundle udlBundle,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
 						   zdl::DlSystem::TensorShapeMap inputDimensions,
                                                   bool useCaching)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessor(runtime)
       .setUdlBundle(udlBundle)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .setPlatformConfig(platformConfig)
       .setInitCacheMode(useCaching)
       .setInputDimensions(inputDimensions)	
       .build();
    return snpe;
}
