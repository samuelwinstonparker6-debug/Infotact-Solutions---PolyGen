import { Canvas, useLoader } from '@react-three/fiber'
import { OrbitControls, Grid, Center } from '@react-three/drei'
import { Suspense, useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'

function LoadedObjMesh({ url, wireframe }) {
  const rawObj = useLoader(OBJLoader, url)

  const styledGroup = useMemo(() => {
    const clone = rawObj.clone(true)
    clone.traverse((child) => {
      if (child.isMesh) {
        if (!child.geometry.attributes.normal) {
          child.geometry.computeVertexNormals()
        }
        child.castShadow = true
        child.receiveShadow = true
        child.material = new THREE.MeshStandardMaterial({
          color: '#cbd5e1',
          roughness: 0.45,
          metalness: 0.25,
          wireframe,
        })
      }
    })
    return clone
  }, [rawObj, wireframe])

  useEffect(() => {
    styledGroup.traverse((child) => {
      if (child.isMesh && child.material) {
        child.material.wireframe = wireframe
      }
    })
  }, [styledGroup, wireframe])

  return (
    <Center>
      <primitive object={styledGroup} />
    </Center>
  )
}

function PlaceholderObjPreview({ wireframe }) {
  return (
    <Center>
      <mesh castShadow receiveShadow>
        <torusKnotGeometry args={[0.75, 0.24, 128, 32]} />
        <meshStandardMaterial
          color="#6366f1"
          roughness={0.35}
          metalness={0.3}
          wireframe={wireframe}
        />
      </mesh>
    </Center>
  )
}

export default function ObjViewer({ modelUrl, wireframe = false, autoRotate = true }) {
  return (
    <Canvas
      shadows
      camera={{ position: [3.2, 2.2, 3.8], fov: 45 }}
      style={{ width: '100%', height: '100%' }}
    >
      {/* Basic 3-Point Scene Lighting */}
      <ambientLight intensity={0.45} />
      <hemisphereLight args={['#e0f2fe', '#0f172a', 0.6]} />
      <directionalLight
        position={[5, 8, 5]}
        intensity={1.6}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <directionalLight position={[-4, 3, -5]} intensity={0.65} color="#38bdf8" />

      <Suspense fallback={null}>
        {modelUrl ? (
          <LoadedObjMesh key={modelUrl} url={modelUrl} wireframe={wireframe} />
        ) : (
          <PlaceholderObjPreview wireframe={wireframe} />
        )}
      </Suspense>

      <Grid
        position={[0, -1.25, 0]}
        args={[12, 12]}
        cellColor="#1e293b"
        sectionColor="#334155"
        fadeDistance={16}
      />

      <OrbitControls makeDefault autoRotate={autoRotate} autoRotateSpeed={1.4} />
    </Canvas>
  )
}

