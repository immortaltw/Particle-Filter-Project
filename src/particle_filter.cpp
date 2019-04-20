/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include "map.h"

using std::normal_distribution;
using std::string;
using std::unordered_map;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  is_initialized = true;

  // Add Gaussian noise to x, y, theta
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 100;

  for (int i=0; i<num_particles; ++i) {
    Particle p = {.id=i, .x=dist_x(gen), .y=dist_y(gen), .theta=dist_theta(gen), .weight=1.0};
    particles.push_back(p);
    weights.push_back(p.weight);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;
  for (auto &p: particles) {
    double xp, yp, thetap;
    if (fabs(yaw_rate) < epsilon) {
      xp = p.x + velocity * delta_t * cos(p.theta);
      yp = p.y + velocity * delta_t * sin(p.theta);
      thetap = p.theta;
    } else {
      xp = p.x + velocity/yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      yp = p.y + velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      thetap = p.theta + yaw_rate * delta_t;
    }
    normal_distribution<double> dist_x(xp, std_pos[0]);
    normal_distribution<double> dist_y(yp, std_pos[1]);
    normal_distribution<double> dist_theta(thetap, std_pos[2]);

    // Reassign x, y, theta to particles with random Gaussian noise.
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);  
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double weight_normalizer = 0.0;

  for (auto &p: particles) {
    vector<LandmarkObs> in_range_landmarks;
    vector<LandmarkObs> obs_in_map_coord;
    auto landmarks = map_landmarks.landmark_list;
    auto landmark_map = map_landmarks.landmark_map;

    // Filter out out of range landmarks.
    for (auto l: landmarks) {
      if (dist(p.x, p.y, l.x_f, l.y_f) <= sensor_range) {
      // if ((fabs((p.x - l.x_f)) <= sensor_range) && (fabs((p.y - l.y_f)) <= sensor_range)) {
        in_range_landmarks.push_back(LandmarkObs{ l.id_i, l.x_f, l.y_f });
      }
    }
  
    // Transform observations from vehicle coord to map coord.
    for (auto o: observations) {
      obs_in_map_coord.push_back(transform_obs(p.x, p.y, p.theta, o));
    }

    // Reset particle weight.
    p.weight = 1.0;

    // Associate landmarks with observations and calculate weight.
    for (auto &om: obs_in_map_coord) {
      double minDist = std::numeric_limits<double>::max();
      double curDist = 0;
      int associate_id = -1;

      for (auto &l: in_range_landmarks) {
        curDist = dist(om.x, om.y, l.x, l.y);
        if (curDist <= minDist) {
          associate_id = l.id;
          minDist = curDist;
        }
      }
      om.id = associate_id;
      double w = multiv_prob(std_landmark[0],
                             std_landmark[1],
                             om.x,
                             om.y,
                             landmark_map[associate_id].x_f,
                             landmark_map[associate_id].y_f);
      if (w == 0) p.weight *= this->epsilon;
      else p.weight *= w;
    }
    weight_normalizer += p.weight;
  }

  // Normalize weights.
  for (int i=0; i<num_particles; ++i) {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> resampled;
  std::default_random_engine gen;
  std::discrete_distribution<double> d(weights.begin(), weights.end());
  for (int i=0; i<particles.size(); ++i) {
    int idx = d(gen);
    resampled.push_back(particles[idx]);
  }
  particles = resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}