package main

import (
	"fmt"
	// "math"
	"math/rand"
	"time"

	"github.com/parallelo-ai/kmeans"
)

// Coordinates is a slice of float64
type Coordinates []float64

// Coordinates implements the Observation interface for a plain set of float64
// coordinates
func (c Coordinates) Coordinates() kmeans.Coordinates {
	return kmeans.Coordinates(c)
}

// Distance returns the euclidean distance between two coordinates
func (c Coordinates) Distance(p2 kmeans.Coordinates) float64 {
	var r float64
	for i, v := range c {
		r += v - p2[i]
		// r += math.Abs(v - p2[i])
		// r += math.Pow(v-p2[i], 2)
	}
	return r
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// set up a random two-dimensional data set (float64 values between 0.0 and 1.0)
	var d1 kmeans.Observations
	var d2 kmeans.Observations
	for x := 0; x < 1024; x++ {
		a := rand.Float64()
		b := rand.Float64()

		// type kmeans Coordinates
		pair1 := kmeans.Coordinates{
			a,
			b,
		}

		// plain type Coordinates
		pair2 := Coordinates{
			a,
			b,
		}

		d1 = append(d1, pair1)
		d2 = append(d2, pair2)
	}
	fmt.Printf("%d data points\n", len(d1))

	// Partition the data points into 7 clusters
	km1 := kmeans.New()
	km2 := kmeans.New()
	clusters1, _ := km1.Partition(d1, 7, 123)
	clusters2, _ := km2.Partition(d2, 7, 123)

	fmt.Println("First clustering")

	for i, c := range clusters1 {
		fmt.Printf("Cluster: %d\n", i)
		fmt.Printf("Centered at x: %.2f y: %.2f\n", c.Center[0], c.Center[1])
	}

	fmt.Println("Second clustering")

	for i, c := range clusters2 {
		fmt.Printf("Cluster: %d\n", i)
		fmt.Printf("Centered at x: %.2f y: %.2f\n", c.Center[0], c.Center[1])
	}

}
