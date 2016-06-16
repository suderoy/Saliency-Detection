/*
    Author @ Sudeshna Roy
    Published in VISAPP 2014 under the title "Saliency Detection using Graph-based Rarity, Spatial Compactness and Background Prior".
    This file modified from a the software originally written by Philipp Krähenbühl, 2012
    All rights reserved.
*/

#include "saliency.h"
#include "fastmath.h"
#include <filter/filter.h>

#define KUpper 4
#define KLower 1

SaliencySettings::SaliencySettings() {
	// Superpixel settings
	n_superpixels_ = 400;
	n_iterations_= 5;
	superpixel_color_weight_ = 1;
	
	// Saliency filter radii
	sigma_p_ = 0.25;
	sigma_c_ = 20.0;
	k_ = 3;
	
	// Weightage
	k_bg_ = 0.5; //experimentally found optimal


	// Upsampling parameters
	min_saliency_ = 0.1;
	alpha_ = 1.0 / 30.0;
	beta_ = 1.0 / 30.0;
	
	// Various algorithm settings
	upsample_ = true;
	use_spix_color_ = false; // Disabled to get a slightly better performance
	
}


Saliency::Saliency( SaliencySettings settings ): settings_(settings), superpixel_( settings.n_superpixels_, settings.superpixel_color_weight_, settings.n_iterations_ ) {
}
Mat_< float > Saliency::saliency( const Mat_< Vec3b >& im ) const {

	// Convert the image to the lab space
	Mat_<Vec3f> rgbim, labim;
	im.convertTo( rgbim, CV_32F, 1.0/255.0 );
	cvtColor( rgbim, labim, CV_BGR2Lab );
	
	// Do the abstraction
	Mat_<int> segmentation = superpixel_.segment( labim );
	std::vector< SuperpixelStatistic > stat = superpixel_.stat( labim, im, segmentation );
	
	// Compute the distribution
	std::vector<float> dist( stat.size(), 0 );
	dist = distribution( stat );

	// Compute spectral rarity
	std::vector<float> rarity( stat.size() , 0 );
	rarity = spectralRarity( stat );
		
	
//	//to save spatial compactness result
//	std::vector<float> s_1( stat.size() );
//	for( int i=0; i<stat.size(); i++ ){
//		s_1[i] = exp( - settings_.k_ * dist[i] );
//	}
//	Mat_<float> s1;
//	s1 = assign( segmentation, s_1 );
//	return s1;

//	//to save F (eqn. 4) result
//	std::vector<float> s_1( stat.size() );
//	for( int i=0; i<stat.size(); i++ ){
//		s_1[i] = rarity[i] * exp( - settings_.k_ * dist[i] );
//	}
//	Mat_<float> s1;
//	s1 = assign( segmentation, s_1 );
//	return s1;
	
	//Compute color distance from boundary (Background Prior)
	std::vector<float> boundaryColor_dist_( stat.size(), 0);
	boundaryColor_dist_ = distinctFromBoundary( stat , segmentation );
	
//	// to save intermediate Boundary Prior result
//	Mat_<float> d;
//	d = assign( segmentation, boundaryColor_dist_ );
//	return d;

	// Combine the measures
	std::vector<float> sp_saliency( stat.size() );
	
	for( int i=0; i<stat.size(); i++ ){
			
		sp_saliency[i] =  ( 1 - settings_.k_bg_ ) * ( rarity[i] * exp( - settings_.k_ * dist[i] ) ) + ( settings_.k_bg_ * boundaryColor_dist_ [ i ] );
	}
	
	Mat_<float> r;
	// Upsampling
	if (settings_.upsample_)
		r = assignFilter( im, segmentation, stat, sp_saliency );
	else
		r = assign( segmentation, sp_saliency );
	
	// Rescale the saliency to [0..1]
	double mn, mx;
	minMaxLoc( r, &mn, & mx );
	r = (r - mn) / (mx - mn);
	
	// Increase the saliency value until we are below the minimal threshold
	double m_sal = settings_.min_saliency_ * r.size().area();
	for( float sm = sum( r )[0]; sm < m_sal; sm = sum( r )[0] )
		r =  min( r*m_sal/sm, 1.0f );
    	
	return r;
	
}
// Normalize a vector of floats to the range [0..1]
void normVec( std::vector< float > &r ){
	const int N = r.size();
	float mn = r[0], mx = r[0];
	for( int i=1; i<N; i++ ) {
		if (mn > r[i])
			mn = r[i];
		if (mx < r[i])
			mx = r[i];
	}
	for( int i=0; i<N; i++ )
		r[i] = (r[i] - mn) / (mx - mn);
}

Mat vectorSubstraction(vector<float> vec1, vector<float> vec2)
{
	Mat mat = Mat(vec1.size(),1,CV_32FC1);
	for (int i=0; i < vec1.size(); i++)
		mat.at<float>(i,0)=vec1.at(i)-vec2.at(i);
	return mat;
}

//Spatial Variance as in eqn. (2)
std::vector< float > Saliency::distribution( const std::vector< SuperpixelStatistic >& stat ) const {
	const int N = stat.size();
	std::vector< float > r( N );
	const float sc =  0.5 / (settings_.sigma_c_*settings_.sigma_c_);
    
    	//can be parallalized using OpenMP
	//#pragma omp parallel for num_threads(2)
	for( int i=0; i<N; i++ ) {
		float u = 0, norm = 1e-10;
		Vec3f c = stat[i].mean_color_;
		Vec2f p(0.f, 0.f);
		
		// Find the mean position
		for( int j=0; j<N; j++ ) {
			Vec3f dc = stat[j].mean_color_ - c;
			float w = fast_exp( - sc * dc.dot(dc) );
			p += w*stat[j].mean_position_;
			norm += w;
		}
		p *= 1.0 / norm;
		
		// Compute the variance
		for( int j=0; j<N; j++ ) {
			Vec3f dc = stat[j].mean_color_ - c;
			Vec2f dp = stat[j].mean_position_ - p;
			
			float w = fast_exp( - sc * dc.dot(dc));
						
			u += w*dp.dot(dp);
			
		}
		r[i] = u / norm;
	}
	normVec( r );
	return r;
}

//Graph based spectral rarity as in eqn. (1)
std::vector< float > Saliency::spectralRarity( const std::vector< SuperpixelStatistic >& stat ) const {
    	
    	int N = stat.size();
    	float sc = 1;
	Mat_<float> A(N, N, CV_32FC1);
	for( int i=0; i<N; i++ ){
		for( int j=0; j<N; j++ ) {
			Vec3f dc = stat[ i ].mean_color_ - stat[ j ].mean_color_;
			A(i,j) = exp( - sc * dc.dot(dc) );
		}
		A(i,i) = 0;
	}
	
	// compute the Laplacian of the adjacency matrix
	Mat_<float> L (N, N),D;
	reduce(A, D, 0, CV_REDUCE_SUM);
	Mat_<float> I = Mat_<float>::eye(N, N);
	D = Mat::diag(D);
	for ( int i=0; i<N; i++ ){
		D(i,i) = 1/sqrt( D(i,i) );
	}
	
	//L = I - (D * A * D);
	for ( int i = 0; i < N ; i++ ){
		for (int j =0 ; j< N ; j++ ){
			L(i,j) = A(i,j)/( D(i,i) * D(j,j) );
		}
	}
	L = I - L;
		
	// calculate the eigenvalues and eigenvectors
	Mat_<float> V, v;
	eigen(L, v, V);
	
	
	// using the Ng laplacian, normalize the eigenvectors
	Mat_<float> norm;
	pow(V, 2, norm);
	reduce(norm, norm, 0, CV_REDUCE_SUM);
	for (unsigned int k = 0; k < N; ++k) V.row(k) = V.row(k) / norm;
	
	std::vector< float > r( N );
	const float sp = 0.5 / (settings_.sigma_p_ * settings_.sigma_p_);
    
	//#pragma omp parallel for num_threads(2)
	for( int i=0; i<N; i++ ) {
		float u = 0, norm = 1e-10;
		Mat_<float> c = V.col(i);
		
		Vec2f p = stat[i].mean_position_;
		
		// Evaluate the rarity score
		for( int j=0; j<N; j++ ) {
			Mat_<float> dc = V.col(j) - c;
			Vec2f dp = stat[j].mean_position_ - p;
			
			
			float w = fast_exp( - sp * dp.dot(dp) );
			Mat_<float> d = dc.t() * dc;			
			u += w * d.at<float>(0,0);
			norm += w;
		}
        
		r[i] = u;
	}
	normVec( r );
	return r;
}

//Background Prior as in eqn. (5)
std::vector<float> Saliency::distinctFromBoundary( const std::vector< SuperpixelStatistic >& stat, const Mat_< int >& segmentation ) const {

	int K = superpixel_.nLabels( segmentation );
	
	int sample_count = 0;
	
    //store all the Lab vectors in a matrix (for all superpixels)
   	cv::Mat allSamples(K, 3, CV_64FC1);
   	
    	for(int i =0; i< K; i++)
    	{
    		allSamples.at< Vec3d > ( i ) = stat[ i ].mean_color_;
    		
    		if(stat [ i ].isOnBoundary == 1 ){
            		sample_count++;
            	}
        }
    		
	//get all the buondary samples in a vector
    	cv::Mat samples(sample_count, 3, CV_32FC1);
   	int idx = 0;
   	for (int k = 0; k < K ; k++){
    		if( stat [ k ].isOnBoundary == 1 ){
          	samples.at< Vec3f > (idx++) = stat[ k ].mean_color_;
     	   }
  	}
    	
	
	//find # of GMM components optimizing k-means compactness
   	int maxCompactAtk = KLower;
    	double maxC = -999999;
	for( int i = KLower ; i<= KUpper ; i++){
	 	int clusterCount = i;
		Mat labels;
		int attempts = 5;
		Mat centers;
		double compactness = kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 0.0001, 10000), attempts, KMEANS_PP_CENTERS, centers );
		
		if( compactness > maxC ){
			maxC = compactness;
			maxCompactAtk = i;
		}
	}
	
	int NrGMMComponents = maxCompactAtk;
	
	//model he oundary segments using GMM
    	Mat likelyhood;
    	Mat labels;
    	Mat probabilities;
    	EM em( NrGMMComponents,  EM::COV_MAT_GENERIC);
   	em.train( samples, likelyhood, labels , probabilities);        

    	//the boundary colors
    	Mat means = em.get<cv::Mat>("means");
    	
    	//the weights of the colors
    	Mat weights = em.get<cv::Mat>("weights");
    	
    	//the cov matrices
    	std::vector<cv::Mat> covs = em.get<std::vector<cv::Mat>>("covs");
    	Mat_<double> covariance = covs [ 0 ];
	
	
	//find distance from the modeled GMMs
	std::vector<float> r( K );
	
	#pragma omp parallel for num_threads(2)
	for (int k = 0; k < K ; k++){
		r[ k ] = 0;
		//get the weighted sum of Mahalonobis ditance from all the GMM components
    
		for(int l = 0 ;l < NrGMMComponents; l++){
					
			Mat_<double> covariance = covs [ l ];
			double mdist_;
			//in case of uniform background, take eucledian dist 
			// Mahalanobis dist would be zero as, cov is a zero mat
			if( determinant(covariance) <= 0 ){
				Vec3d d = means.at<Vec3d>( l ) - allSamples.at<Vec3d>( k );
				mdist_ = d.dot(d);
			}
			else{
				mdist_ = Mahalanobis( means.at<Vec3d>( l ), allSamples.at<Vec3d>( k ), covariance.inv(DECOMP_SVD));
			}

			double weight = weights.at<double> ( l )>0.3?weights.at<double> ( l ):0;
			r[ k ] += mdist_ * weight;
			
		}
		
	}
	normVec( r );
		
	return r;
}

Mat_< float > Saliency::assign( const Mat_< int >& seg, const std::vector< float >& sal ) const {
	return Superpixel::assign( sal, seg );
}

//Up-sampling code as used by Saliency Filter (CVPR 2012) paper, using the concept of 
//J. Dolson, B. Jongmin, C. Plagemann, and S. Thrun, “Upsampling range data in dynamic environments,” in CVPR ,2010, pp. 1141–1148.
Mat_< float > Saliency::assignFilter( const Mat_< Vec3b >& im, const Mat_< int >& seg, const vector< SuperpixelStatistic >& stat, const std::vector< float >& sal ) const {
	std::vector< float > source_features( seg.size().area()*5 ), target_features( im.size().area()*5 );
	Mat_< Vec2f > data( seg.size() );
	// There is a type on the paper: alpha and beta are actually squared, or directly applied to the values
	const float a = settings_.alpha_, b = settings_.beta_;
	
	const int D = 5;
	// Create the source features
	for( int j=0,k=0; j<seg.rows; j++ )
		for( int i=0; i<seg.cols; i++, k++ ) {
			int id = seg(j,i);
			data(j,i) = Vec2f( sal[id], 1 );
			
			source_features[D*k+0] = a * i;
			source_features[D*k+1] = a * j;
			if (D == 5) {
				source_features[D*k+2] = b * stat[id].mean_rgb_[0];
				source_features[D*k+3] = b * stat[id].mean_rgb_[1];
				source_features[D*k+4] = b * stat[id].mean_rgb_[2];
			}
		}
	// Create the source features
	for( int j=0,k=0; j<im.rows; j++ )
		for( int i=0; i<im.cols; i++, k++ ) {
			target_features[D*k+0] = a * i;
			target_features[D*k+1] = a * j;
			if (D == 5) {
				target_features[D*k+2] = b * im(j,i)[0];
				target_features[D*k+3] = b * im(j,i)[1];
				target_features[D*k+4] = b * im(j,i)[2];
			}
		}
	
	// Do the filtering [Filtering using the target features twice works slightly better, as the method described in our paper]
	if (settings_.use_spix_color_) {
		Filter filter( source_features.data(), seg.cols*seg.rows, target_features.data(), im.cols*im.rows, D );
		filter.filter( data.ptr<float>(), data.ptr<float>(), 2 );
	}
	else {
		Filter filter( target_features.data(), im.cols*im.rows, D );
		filter.filter( data.ptr<float>(), data.ptr<float>(), 2 );
	}
	
	Mat_<float> r( im.size() );
	for( int j=0; j<im.rows; j++ )
		for( int i=0; i<im.cols; i++ )
			r(j,i) = data(j,i)[0] / (data(j,i)[1] + 1e-10);
	return r;
}






