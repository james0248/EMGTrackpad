import prand, { type RandomGenerator } from "pure-rand";

/**
 * Seeded pseudo-random number generator using pure-rand (xoroshiro128+)
 * Wraps pure-rand's immutable API in a mutable class for convenience.
 */
export class PRNG {
  private rng: RandomGenerator;

  constructor(seed: number) {
    this.rng = prand.xoroshiro128plus(seed);
  }

  /**
   * Returns a random float in [0, 1)
   */
  next(): number {
    const [value, nextRng] = prand.uniformIntDistribution(0, 0x7fffffff, this.rng);
    this.rng = nextRng;
    return value / 0x80000000;
  }

  /**
   * Returns a random integer in [min, max] inclusive
   */
  nextInt(min: number, max: number): number {
    const [value, nextRng] = prand.uniformIntDistribution(min, max, this.rng);
    this.rng = nextRng;
    return value;
  }

  /**
   * Returns a random float in [min, max)
   */
  nextFloat(min: number, max: number): number {
    return this.next() * (max - min) + min;
  }

  /**
   * Returns a random boolean with given probability of true
   */
  nextBool(probability = 0.5): boolean {
    return this.next() < probability;
  }

  /**
   * Picks a random element from an array
   */
  pick<T>(arr: T[]): T {
    return arr[this.nextInt(0, arr.length - 1)]!;
  }

  /**
   * Shuffles an array in place (Fisher-Yates)
   */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.nextInt(0, i);
      [arr[i], arr[j]] = [arr[j]!, arr[i]!];
    }
    return arr;
  }
}

/**
 * Generate a random seed based on current time
 */
export function generateSeed(): number {
  return Math.floor(Math.random() * 0x7fffffff);
}
