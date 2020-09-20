//==============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SETBUILDEROPTIONS_H
#define SETBUILDEROPTIONS_H

#include "SNPE/SNPE.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/UDLFunc.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/PlatformConfig.hpp"
#include "DlSystem/TensorShapeMap.hpp"

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::UDLBundle udlBundle,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching);
std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::UDLBundle udlBundle,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
 						   zdl::DlSystem::TensorShapeMap inputDimensions,
                                                   bool useCaching);

#endif
