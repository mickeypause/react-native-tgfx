import { createContext, type ReactNode } from 'react'
import ReactReconciler from 'react-reconciler'
import {
  ConcurrentRoot,
  DefaultEventPriority,
  NoEventPriority,
} from 'react-reconciler/constants'
import { TgfxContainer, type CommitListener } from './Container'
import { createNode, type Node, type NodeTypeName } from './types'

type Instance = Node
type Container = TgfxContainer

let currentUpdatePriority: number = NoEventPriority

// Non-null, or react-reconciler warns "Expected host context to exist".
const HOST_CONTEXT = {}

const HostTransitionContext = createContext<unknown>(null)

// Called per cloned node per commit, so it counts keys in place rather than allocating two arrays.
export function shallowEqualProps(
  a: Record<string, unknown>,
  b: Record<string, unknown>
): boolean {
  for (const key in a) {
    if (a[key] !== b[key]) {
      return false
    }
  }
  for (const key in b) {
    if (!(key in a)) {
      return false
    }
  }
  return true
}

const hostConfig = {
  supportsMutation: false,
  supportsPersistence: true,
  supportsHydration: false,
  isPrimaryRenderer: false,
  noTimeout: -1 as const,
  scheduleTimeout: setTimeout,
  cancelTimeout: clearTimeout,

  createInstance(type: NodeTypeName, props: Record<string, unknown>): Instance {
    return createNode(type, props)
  },
  createTextInstance(text: string): Instance {
    return createNode('tgfx_rawtext' as NodeTypeName, { text })
  },
  appendInitialChild(parent: Instance, child: Instance): void {
    parent.children.push(child)
  },
  finalizeInitialChildren(): boolean {
    return false
  },
  shouldSetTextContent(): boolean {
    return false
  },
  getRootHostContext(): object {
    return HOST_CONTEXT
  },
  getChildHostContext(): object {
    return HOST_CONTEXT
  },
  getPublicInstance(instance: Instance): Instance {
    return instance
  },
  prepareForCommit(): null {
    return null
  },
  resetAfterCommit(container: Container): void {
    container.redraw()
  },
  preparePortalMount(): void {},

  cloneInstance(
    instance: Instance,
    type: NodeTypeName,
    oldProps: Record<string, unknown>,
    newProps: Record<string, unknown>,
    keepChildren: boolean
  ): Instance {
    if (keepChildren && shallowEqualProps(oldProps, newProps)) {
      return instance
    }
    return {
      type,
      props: newProps,
      children: keepChildren ? [...instance.children] : [],
    }
  },
  cloneHiddenInstance(
    instance: Instance,
    type: NodeTypeName,
    props: Record<string, unknown>
  ): Instance {
    return { type, props, children: [...instance.children] }
  },
  cloneHiddenTextInstance(instance: Instance): Instance {
    return instance
  },
  createContainerChildSet(): Instance[] {
    return []
  },
  appendChildToContainerChildSet(childSet: Instance[], child: Instance): void {
    childSet.push(child)
  },
  finalizeContainerChildren(): void {},
  replaceContainerChildren(
    container: Container,
    newChildren: Instance[]
  ): void {
    container.root = newChildren
  },
  clearContainer(container: Container): void {
    container.root = []
  },

  getInstanceFromNode: () => null,
  beforeActiveInstanceBlur: () => {},
  afterActiveInstanceBlur: () => {},
  prepareScopeUpdate: () => {},
  getInstanceFromScope: () => null,
  detachDeletedInstance: () => {},

  setCurrentUpdatePriority(priority: number): void {
    currentUpdatePriority = priority
  },
  getCurrentUpdatePriority(): number {
    return currentUpdatePriority
  },
  resolveUpdatePriority(): number {
    return currentUpdatePriority !== NoEventPriority
      ? currentUpdatePriority
      : DefaultEventPriority
  },

  NotPendingTransition: null,
  HostTransitionContext,
  resetFormInstance: () => {},
  requestPostPaintCallback: () => {},
  shouldAttemptEagerTransition: () => false,
  trackSchedulerEvent: () => {},
  resolveEventType: () => null,
  resolveEventTimeStamp: () => -1,
  maySuspendCommit: () => false,
  preloadInstance: () => true,
  startSuspendingCommit: () => {},
  suspendInstance: () => {},
  waitForCommitToBeReady: () => null,
}

// The full HostConfig type is enormous, so this provides what React 19 invokes and casts.
const reconciler = ReactReconciler(
  hostConfig as unknown as Parameters<typeof ReactReconciler>[0]
)

/** A mounted reconciler root that renders a React element tree into a container. */
export class TgfxRoot {
  readonly container: TgfxContainer
  private readonly root: ReturnType<typeof reconciler.createContainer>
  private direct = false

  constructor(onCommit: CommitListener) {
    this.container = new TgfxContainer(onCommit)
    this.root = reconciler.createContainer(
      this.container,
      ConcurrentRoot,
      null,
      false,
      null,
      '',
      (error: Error) => console.error('[tgfx] uncaught error:', error),
      (error: Error) => console.error('[tgfx] caught error:', error),
      (error: Error) => console.error('[tgfx] recoverable error:', error),
      () => {}
    )
  }

  render(element: ReactNode): void {
    this.direct = false
    reconciler.updateContainer(element, this.root, null, null)
  }

  /// Commits before returning, so a caller with no view can read the program out.
  renderSync(element: ReactNode): void {
    this.direct = false
    reconciler.updateContainerSync(element, this.root, null, null)
    reconciler.flushSyncWork()
  }

  setDirect(roots: Node[]): void {
    if (!this.direct) {
      // `updateContainer` only schedules, so flush it muted or the teardown blanks the canvas.
      this.container.muted = true
      try {
        reconciler.updateContainerSync(null, this.root, null, null)
        reconciler.flushSyncWork()
      } finally {
        this.container.muted = false
      }
      this.direct = true
    }
    this.container.setRootDirect(roots)
  }

  unmount(): void {
    // Flushed muted, as in setDirect: by now Fabric may have recycled the view
    // into the next canvas, and this teardown's empty scene would blank it.
    this.container.muted = true
    try {
      reconciler.updateContainerSync(null, this.root, null, null)
      reconciler.flushSyncWork()
      if (this.container.root.length > 0) {
        this.container.setRootDirect([])
      }
    } finally {
      this.container.muted = false
    }
  }
}
