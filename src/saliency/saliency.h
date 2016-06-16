/*
 Author @ Sudeshna Roy
 Published in VISAPP 2014 under the title "Saliency Detection using Graph-based Rarity, Spatial Compactness and Background Prior".
 This file modified from a the software originally written by Philipp Krähenbühl, 2012
 All rights reserved.
 */

#pragma once
#include "superpixel/superpixel.h"

struct SaliencySettings{
	SaliencySettings();
	
	// Superpixel settings
	int n_superpixels_, n_iterations_;
	float superpixel_color_weight_;
	
	// Saliency filter radii
	float sigma_p_; // Radius for the rarity operator [eq 1]
	float sigma_c_; // Color radius for the spatial variance operator [eq 2]
	float k_; // The sharpness parameter of the exponential in spatial compactness [eq 4]
	
	// Weightage
	float k_bg_; // Fraction weightage for Discrimination from boundary patches - equal weightage is given in the paper
		
	// Upsampling parameters
	float min_saliency_; // Minimum number of salient pixels for final rescaling
	float alpha_, beta_;
	
	// Various algorithm settings
	// Enable or disable parts of the algorithm
	bool upsample_;
	// Should we use the image color or superpixel color as a feature for upsampling
	bool use_spix_color_;

};

class Saliency {
protected:
	SaliencySettings settings_;
	Superpixel superpixel_;
protected:
	std::vector< float > distribution( const std::vector< SuperpixelStatistic > & stat ) const;
	std::vector< float > distinctFromBoundary( const std::vector< SuperpixelStatistic > & stat, const Mat_<int> & seg ) const;
	std::vector< float > spectralRarity( const std::vector< SuperpixelStatistic >& stat ) const;
	std::vector<float> distinctFromBoundarySpectralCluster( const std::vector< SuperpixelStatistic >& stat, const Mat_< int >& segmentation ) const;
	Mat_<float> assign( const Mat_<int> & seg, const std::vector< float > & sal ) const;
	
	Mat_<float> assignFilter( const Mat_<Vec3b> & im, const Mat_<int> & seg, const std::vector< SuperpixelStatistic > & stat, const std::vector< float > & sal ) const ;
	
public:
	Saliency( SaliencySettings settings = SaliencySettings() );
	Mat_<float> saliency( const Mat_<Vec3b> & im ) const;
	
};
